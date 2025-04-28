# Copyright (c) 2024-2025, NVIDIA CORPORATION. All rights reserved.
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
            "RemoteFile not available, please build KvikIO "
            "with libcurl (-DKvikIO_REMOTE_SUPPORT=ON)"
        )
    import kvikio._lib.remote_handle

    return kvikio._lib.remote_handle


class RemoteFile:
    """File handle of a remote file."""

    def __init__(self, handle):
        """Create a remote file from a Cython handle.

        This constructor should not be called directly instead use a
        factory method like `RemoteFile.open_http()`

        Parameters
        ----------
        handle : kvikio._lib.remote_handle.RemoteFile
            The Cython handle
        """
        assert isinstance(handle, _get_remote_module().RemoteFile)
        self._handle = handle

    @classmethod
    def open_http(
        cls,
        url: str,
        nbytes: Optional[int] = None,
    ) -> RemoteFile:
        """Open a http file.

        Parameters
        ----------
        url
            URL to the remote file.
        nbytes
            The size of the file. If None, KvikIO will ask the server
            for the file size.
        """
        return RemoteFile(_get_remote_module().RemoteFile.open_http(url, nbytes))

    @classmethod
    def open_s3(
        cls,
        bucket_name: str,
        object_name: str,
        nbytes: Optional[int] = None,
    ) -> RemoteFile:
        """Open a AWS S3 file from a bucket name and object name.

        Please make sure to set the AWS environment variables:
          - `AWS_DEFAULT_REGION`
          - `AWS_ACCESS_KEY_ID`
          - `AWS_SECRET_ACCESS_KEY`
          - `AWS_SESSION_TOKEN` (when using temporary credentials)

        Additionally, to overwrite the AWS endpoint, set `AWS_ENDPOINT_URL`.
        See <https://docs.aws.amazon.com/cli/v1/userguide/cli-configure-envvars.html>

        Parameters
        ----------
        bucket_name
            The bucket name of the file.
        object_name
            The object name of the file.
        nbytes
            The size of the file. If None, KvikIO will ask the server
            for the file size.
        """
        return RemoteFile(
            _get_remote_module().RemoteFile.open_s3(bucket_name, object_name, nbytes)
        )

    @classmethod
    def open_s3_url(
        cls,
        url: str,
        nbytes: Optional[int] = None,
    ) -> RemoteFile:
        """Open a AWS S3 file from an URL.

        The `url` can take two forms:
          - A full http url such as "http://127.0.0.1/my/file", or
          - A S3 url such as "s3://<bucket>/<object>".

        Please make sure to set the AWS environment variables:
          - `AWS_DEFAULT_REGION`
          - `AWS_ACCESS_KEY_ID`
          - `AWS_SECRET_ACCESS_KEY`
          - `AWS_SESSION_TOKEN` (when using temporary credentials)

        Additionally, if `url` is a S3 url, it is possible to overwrite the AWS endpoint
        by setting `AWS_ENDPOINT_URL`.
        See <https://docs.aws.amazon.com/cli/v1/userguide/cli-configure-envvars.html>

        Parameters
        ----------
        url
            Either a http url or a S3 url.
        nbytes
            The size of the file. If None, KvikIO will ask the server
            for the file size.
        """
        url = url.lower()
        if url.startswith("http://") or url.startswith("https://"):
            return RemoteFile(
                _get_remote_module().RemoteFile.open_s3_from_http_url(url, nbytes)
            )
        if url.startswith("s3://"):
            return RemoteFile(
                _get_remote_module().RemoteFile.open_s3_from_s3_url(url, nbytes)
            )
        raise ValueError(f"Unsupported protocol: {url}")

    def close(self) -> None:
        """Close the file"""
        pass

    def __enter__(self) -> RemoteFile:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def __str__(self) -> str:
        return str(self._handle)

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
