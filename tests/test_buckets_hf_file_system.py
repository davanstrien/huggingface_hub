import os
import threading
from unittest.mock import Mock, patch

import pytest

from huggingface_hub.hf_file_system import HfFileSystem

from .test_hf_file_system import _HfFileSystemBaseROTests, _HfFileSystemBaseRWTests, _HfFileSystemBucketChecks
from .testing_constants import ENDPOINT_STAGING, TOKEN
from .testing_utils import repo_name


pytestmark = pytest.mark.xet


class TestHfFileSystemBucketRO(_HfFileSystemBucketChecks, _HfFileSystemBaseROTests):
    __test__ = True

    @pytest.fixture(scope="class", autouse=True)
    def _shared_bucket(self, request):
        api = type(self).api

        # Create dummy bucket
        repo_url = api.create_bucket(repo_name())
        bucket_id = repo_url.bucket_id
        hf_path = f"buckets/{bucket_id}"
        request.cls.bucket_id = bucket_id
        request.cls.hf_path = hf_path

        # Upload files
        api.batch_bucket_files(
            bucket_id,
            add=[
                ("dummy text data".encode("utf-8"), "data/text_data.txt"),
                (b"dummy binary data", "data/binary_data.bin"),
                ("# Dataset card".encode("utf-8"), "README.md"),
            ],
        )

        request.cls.readme_file_path = "README.md"
        request.cls.readme_file = hf_path + "/" + "README.md"
        request.cls.text_file_path = "data/text_data.txt"
        request.cls.text_file = hf_path + "/" + "data/text_data.txt"
        yield
        api.delete_bucket(bucket_id)

    @pytest.fixture(autouse=True)
    def _new_hffs(self):
        self.hffs = HfFileSystem(endpoint=ENDPOINT_STAGING, token=TOKEN, skip_instance_cache=True)

    def test_prefix_collision(self):
        colliding_path = f"{self.hf_path}/dat"

        assert not self.hffs.exists(f"{colliding_path}/new.txt")
        assert self.hffs.glob(f"{colliding_path}/*") == []
        with pytest.raises(FileNotFoundError):
            self.hffs.ls(colliding_path)


class TestHfFileSystemBucketRW(_HfFileSystemBucketChecks, _HfFileSystemBaseRWTests):
    __test__ = True

    @pytest.fixture(autouse=True)
    def _bucket(self):
        self.hffs = HfFileSystem(endpoint=ENDPOINT_STAGING, token=TOKEN, skip_instance_cache=True)

        # Create dummy bucket
        repo_url = self.api.create_bucket(repo_name())
        self.bucket_id = repo_url.bucket_id
        self.hf_path = f"buckets/{self.bucket_id}"

        # Upload files
        self.api.batch_bucket_files(
            self.bucket_id,
            add=[
                ("dummy text data".encode("utf-8"), "data/text_data.txt"),
                (b"dummy binary data", "data/binary_data.bin"),
                ("# Dataset card".encode("utf-8"), "README.md"),
            ],
        )

        self.readme_file_path = "README.md"
        self.readme_file = self.hf_path + "/" + self.readme_file_path
        self.text_file_path = "data/text_data.txt"
        self.text_file = self.hf_path + "/" + self.text_file_path
        yield
        self.api.delete_bucket(self.bucket_id)

    @pytest.mark.skip("Not implemented yet")
    def test_copy_file(self):
        pass

    def _assert_text_file_unchanged(self):
        self.hffs.invalidate_cache()
        assert self.hffs.cat_file(self.text_file) == b"dummy text data"

    def test_write_does_not_use_local_temp_file(self):
        data = os.urandom(12 << 20)  # 12MB => several 5MB blocks
        with patch("huggingface_hub.hf_file_system.tempfile.NamedTemporaryFile", side_effect=AssertionError):
            with self.hffs.open(self.hf_path + "/data/small.bin", "wb") as f:
                f.write(b"small")  # single (final) chunk
            with self.hffs.open(self.hf_path + "/data/big.bin", "wb") as f:
                for start in range(0, len(data), 1 << 20):
                    f.write(data[start : start + (1 << 20)])
        assert self.hffs.cat_file(self.hf_path + "/data/small.bin") == b"small"
        assert self.hffs.info(self.hf_path + "/data/big.bin")["size"] == len(data)
        assert self.hffs.cat_file(self.hf_path + "/data/big.bin") == data

    def test_write_empty_file(self):
        self.api.batch_bucket_files(self.bucket_id, add=[(b"", "empty_from_api.bin")])
        with self.hffs.open(self.hf_path + "/empty_from_fs.bin", "wb"):
            pass
        expected = self.hffs.info(self.hf_path + "/empty_from_api.bin")
        info = self.hffs.info(self.hf_path + "/empty_from_fs.bin")
        assert info["size"] == expected["size"] == 0
        assert info["xet_hash"] == expected["xet_hash"]
        assert self.hffs.cat_file(self.hf_path + "/empty_from_fs.bin") == b""

    def test_write_overwrites_existing_file(self):
        with self.hffs.open(self.text_file, "wb") as f:
            f.write(b"new content")
        assert self.hffs.cat_file(self.text_file) == b"new content"
        assert self.hffs.info(self.text_file)["size"] == len(b"new content")

    def test_write_concurrently(self):
        contents = {f"data/thread_{i}.bin": os.urandom(6 << 20) for i in range(2)}
        errors = []

        def _write(path: str, data: bytes) -> None:
            try:
                with self.hffs.open(f"{self.hf_path}/{path}", "wb") as f:
                    for start in range(0, len(data), 1 << 20):
                        f.write(data[start : start + (1 << 20)])
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=_write, args=item) for item in contents.items()]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert errors == []
        for path, data in contents.items():
            assert self.hffs.cat_file(f"{self.hf_path}/{path}") == data

    @pytest.mark.parametrize("size", [10, 6 << 20])  # exception raised before / after the upload started
    def test_exception_in_with_block_does_not_commit(self, size: int):
        with pytest.raises(RuntimeError):
            with self.hffs.open(self.text_file, "wb") as f:
                f.write(os.urandom(size))
                raise RuntimeError("writer failed midway")
        assert f.closed
        self._assert_text_file_unchanged()

        with pytest.raises(RuntimeError):
            with self.hffs.open(self.hf_path + "/data/never_written.bin", "wb") as f:
                f.write(os.urandom(size))
                raise RuntimeError("writer failed midway")
        self.hffs.invalidate_cache()
        assert not self.hffs.exists(self.hf_path + "/data/never_written.bin")

    @pytest.mark.parametrize("size", [10, 6 << 20])
    def test_discard_then_close(self, size: int):
        f = self.hffs.open(self.text_file, "wb")
        f.write(os.urandom(size))
        f.discard()
        assert f.closed
        f.discard()  # idempotent
        f.close()
        del f  # garbage collection must not commit either
        self._assert_text_file_unchanged()

    def test_failed_write_is_not_committed_on_close(self):
        f = self.hffs.open(self.text_file, "wb")
        f.write(os.urandom(6 << 20))  # starts the stream
        f._xet_stream = Mock(wraps=f._xet_stream)
        f._xet_stream.write.side_effect = OSError("network error")
        with pytest.raises(OSError, match="network error"):
            f.write(os.urandom(6 << 20))
        assert f.closed
        f._xet_stream.abort.assert_called_once()
        f.close()
        self._assert_text_file_unchanged()

    def test_write_falls_back_to_temp_file_without_xet_stream(self):
        with patch("huggingface_hub.hf_file_system._is_xet_stream_upload_available", return_value=False):
            with self.hffs.open(self.hf_path + "/data/fallback.bin", "wb") as f:
                f.write(b"fallback")
                assert f._xet_stream is None
            assert self.hffs.cat_file(self.hf_path + "/data/fallback.bin") == b"fallback"

            with pytest.raises(RuntimeError):
                with self.hffs.open(self.text_file, "wb") as f:
                    f.write(os.urandom(6 << 20))
                    temp_file_name = f.temp_file.name
                    raise RuntimeError("writer failed midway")
            assert not os.path.exists(temp_file_name)
            self._assert_text_file_unchanged()
