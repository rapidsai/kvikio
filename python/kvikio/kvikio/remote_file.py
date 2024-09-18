# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.

from __future__ import annotations

import functools
from typing import Optional

from kvikio.cufile import IOFuture


@functools.cache
def is_remote_file_available() -> bool:
    """Check if the remote module is available"""
    try:
        import kvikio._lib.remote_handle  # noqa: F401
    except ImportError:
        return False
    else:
        return True


@functools.cache
def _get_remote_module():
    """Get the remote module or raise an error"""
    if not is_remote_file_available():
        raise RuntimeError(
            "RemoteFile not available, please build KvikIO with AWS S3 support"
        )
    import kvikio._lib.remote_handle

    return kvikio._lib.remote_handle


class RemoteFile:
    """File handle of a remote file (currently, only AWS S3 is supported)."""

    def __init__(self, url: str, nbytes: Optional[int] = None):
        """Open a remote file given a bucket and object name.

        Parameters
        ----------
        url
            URL to the remote file.
        """
        self._handle = _get_remote_module().RemoteFile.from_url(url, nbytes)

    def __enter__(self) -> RemoteFile:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        pass

    def nbytes(self) -> int:
        """Get the file size.

        Note, this is very fast, no communication needed.

        Returns
        -------
        The number of bytes.
        """
        return self._handle.nbytes()

    def read(self, buf, size: Optional[int] = None, file_offset: int = 0) -> int:
        """Read from remote source into buffer (host or device memory) in parallel.

        Parameters
        ----------
        buf : buffer-like or array-like
            Device or host buffer to read into.
        size
            Size in bytes to read.
        file_offset
            Offset in the file to read from.

        Returns
        -------
        The size of bytes that were successfully read.
        """
        return self.pread(buf, size, file_offset).get()

    def pread(self, buf, size: Optional[int] = None, file_offset: int = 0) -> IOFuture:
        """Read from remote source into buffer (host or device memory) in parallel.

        Parameters
        ----------
        buf : buffer-like or array-like
            Device or host buffer to read into.
        size
            Size in bytes to read.
        file_offset
            Offset in the file to read from.

        Returns
        -------
        Future that on completion returns the size of bytes that were successfully
        read.
        """
        return IOFuture(self._handle.pread(buf, size, file_offset))
