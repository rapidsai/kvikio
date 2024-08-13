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
    """File handle of a remote file (currently, only AWS S3 is supported).

    Please make sure that AWS credentials have been configure on the system.
    A common way to do this, is to define the set the environment variables:
    `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY`.

    Other relevant options are `AWS_DEFAULT_REGION` and `AWS_ENDPOINT_URL`, see
    <https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-envvars.html>.
    """

    def __init__(self, bucket_name: str, object_name: str):
        """Open a remote file given a bucket and object name.

        Parameters
        ----------
        bucket_name
            Name of the bucket.
        object_name
            Name of the object.
        """
        self._handle = _get_remote_remote_file_class().from_bucket_and_object(
            bucket_name, object_name
        )

    @classmethod
    def from_url(cls, url: str) -> RemoteFile:
        """Open a remote file given an url such as "s3://<bucket>/<object>".

        Parameters
        ----------
        url
            URL to the remote file.

        Returns
        -------
        A newly opened remote file
        """
        ret = object.__new__(cls)
        ret._handle = _get_remote_remote_file_class().from_url(url)
        return ret

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
        buf: buffer-like or array-like
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
        buf: buffer-like or array-like
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
