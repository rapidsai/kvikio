# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.

from __future__ import annotations

from typing import Optional

from kvikio.cufile import IOFuture


def is_remote_file_available() -> bool:
    try:
        import kvikio._lib.remote_handle  # noqa: F401
    except ImportError:
        return False
    else:
        return True


def _get_remote_remote_file_class():
    if not is_remote_file_available():
        raise RuntimeError(
            "RemoteFile not available, please build KvikIO with AWS S3 support"
        )
    import kvikio._lib.remote_handle

    return kvikio._lib.remote_handle.RemoteFile


class RemoteFile:
    """File handle of a remote file"""

    def __init__(self, bucket_name: str, object_name: str):
        self._handle = _get_remote_remote_file_class().from_bucket_and_object(
            bucket_name, object_name
        )

    @classmethod
    def from_url(cls, url: str) -> RemoteFile:
        ret = object.__new__(cls)
        ret._handle = _get_remote_remote_file_class().from_url(url)
        return ret

    def __enter__(self) -> RemoteFile:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        pass

    def nbytes(self) -> int:
        return self._handle.nbytes()

    def pread(self, buf, size: Optional[int] = None, file_offset: int = 0) -> IOFuture:
        return IOFuture(self._handle.pread(buf, size, file_offset))

    def read(self, buf, size: Optional[int] = None, file_offset: int = 0) -> int:
        return self.pread(buf, size, file_offset).get()
