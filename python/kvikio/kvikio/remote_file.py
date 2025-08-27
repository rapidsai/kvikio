# Copyright (c) 2024-2025, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.

from __future__ import annotations

import enum
import functools
import urllib.parse
from typing import Optional

from kvikio.cufile import IOFuture


class RemoteEndpointType(enum.Enum):
    """
    Types of remote file endpoints supported by KvikIO.

    This enum defines the different protocols and services that can be used
    to access remote files. It is used to specify or detect the type of
    remote endpoint when opening files.

    Attributes
    ----------
    AUTO : int
        Automatically detect the endpoint type from the URL. KvikIO will
        attempt to infer the appropriate protocol based on the URL format.
    S3 : int
        AWS S3 endpoint using credentials-based authentication. Requires
        AWS environment variables (such as AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY,
        AWS_DEFAULT_REGION) to be set.
    S3_PRESIGNED_URL : int
        AWS S3 endpoint using a presigned URL. No credentials required as
        authentication is embedded in the URL with time-limited access.
    WEBHDFS : int
        Apache Hadoop WebHDFS (Web-based Hadoop Distributed File System)
        endpoint for accessing files stored in HDFS over HTTP/HTTPS.
    HTTP : int
        Generic HTTP or HTTPS endpoint for accessing files from web servers.
        This is used for standard web resources that do not fit the other
        specific categories.

    See Also
    --------
    RemoteFile.open : Factory method that uses this enum to specify endpoint types.
    """

    AUTO = 0
    S3 = 1
    S3_PRESIGNED_URL = 2
    WEBHDFS = 3
    HTTP = 4

    @staticmethod
    def _map_to_internal(remote_endpoint_type: RemoteEndpointType):
        return _get_remote_module().RemoteEndpointType[remote_endpoint_type.name]


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
        parsed_result = urllib.parse.urlparse(url.lower())
        if parsed_result.scheme in ("http", "https"):
            return RemoteFile(
                _get_remote_module().RemoteFile.open_s3_from_http_url(url, nbytes)
            )
        if parsed_result.scheme == "s3":
            return RemoteFile(
                _get_remote_module().RemoteFile.open_s3_from_s3_url(url, nbytes)
            )
        raise ValueError(f"Unsupported protocol: {url}")

    @classmethod
    def open_s3_presigned_url(
        cls,
        presigned_url: str,
        nbytes: Optional[int] = None,
    ) -> RemoteFile:
        """Open a AWS S3 file from a presigned URL.

        Parameters
        ----------
        presigned_url
            Presigned URL to the remote file.
        nbytes
            The size of the file. If None, KvikIO will ask the server
            for the file size.
        """
        return RemoteFile(
            _get_remote_module().RemoteFile.open_s3_from_http_presigned_url(
                presigned_url, nbytes
            )
        )

    @classmethod
    def open_webhdfs(
        cls,
        url: str,
        nbytes: Optional[int] = None,
    ) -> RemoteFile:
        """Open a file on Apache Hadoop Distributed File System (HDFS) using WebHDFS.

        If KvikIO is run within a Docker, the argument ``--network host`` needs to be
        passed to the ``docker run`` command.

        Parameters
        ----------
        url
            URL to the remote file.
        nbytes
            The size of the file. If None, KvikIO will ask the server for the file
            size.
        """
        return RemoteFile(_get_remote_module().RemoteFile.open_webhdfs(url, nbytes))

    @classmethod
    def open(
        cls,
        url: str,
        remote_endpoint_type: RemoteEndpointType = RemoteEndpointType.AUTO,
        allow_list: Optional[list] = None,
        nbytes: Optional[int] = None,
    ) -> RemoteFile:
        """
        Create a remote file handle from a URL.

        This function creates a RemoteFile for reading data from various remote
        endpoints including HTTP/HTTPS servers, AWS S3 buckets, S3 presigned URLs,
        and WebHDFS. The endpoint type can be automatically detected from the URL
        or explicitly specified.

        Parameters
        ----------
        url : str
            The URL of the remote file. Supported formats include:

            - S3 with credentials
            - S3 presigned URL
            - WebHDFS
            - HTTP/HTTPS
        remote_endpoint_type : RemoteEndpointType, optional
            The type of remote endpoint. Default is :class:`RemoteEndpointType.AUTO`
            which automatically detects the endpoint type from the URL. Can be
            explicitly set to :class:`RemoteEndpointType.S3`,
            :class:`RemoteEndpointType.S3_PRESIGNED_URL`,
            :class:`RemoteEndpointType.WEBHDFS`, or :class:`RemoteEndpointType.HTTP`
            to force a specific endpoint type.
        allow_list : list of RemoteEndpointType, optional
            List of allowed endpoint types. If provided:

            - If remote_endpoint_type is :class:`RemoteEndpointType.AUTO`, types are
              tried in the exact order specified until a match is found.
            - In explicit mode, the specified type must be in this list, otherwise an
              exception is thrown.

            If not provided, defaults to all supported types in this order:
            :class:`RemoteEndpointType.S3`,
            :class:`RemoteEndpointType.S3_PRESIGNED_URL`,
            :class:`RemoteEndpointType.WEBHDFS`, and :class:`RemoteEndpointType.HTTP`.
        nbytes : int, optional
            File size in bytes. If not provided, the function sends an additional
            request to the server to query the file size.

        Returns
        -------
        RemoteFile
            A RemoteFile object that can be used to read data from the remote file.

        Raises
        ------
        RuntimeError
            - If the URL is malformed or missing required components.
            - :class:`RemoteEndpointType.AUTO` mode is used and the URL does not match
              any supported endpoint type.
            - The specified endpoint type is not in the `allow_list`.
            - The URL is invalid for the specified endpoint type.
            - Unable to connect to the remote server or determine file size
              (when nbytes not provided).

        Examples
        --------
        - Auto-detect endpoint type from URL:

          .. code-block::

             handle = RemoteFile.open(
                 "https://bucket.s3.amazonaws.com/object?X-Amz-Algorithm=AWS4-HMAC-SHA256"
                 "&X-Amz-Credential=...&X-Amz-Signature=..."
             )

        - Open S3 file with explicit endpoint type:

          .. code-block::

             handle = RemoteFile.open(
                 "https://my-bucket.s3.us-east-1.amazonaws.com/data.bin",
                 remote_endpoint_type=RemoteEndpointType.S3
             )

        - Restrict endpoint type candidates:

          .. code-block::

             handle = RemoteFile.open(
                 user_provided_url,
                 remote_endpoint_type=RemoteEndpointType.AUTO,
                 allow_list=[
                     RemoteEndpointType.HTTP,
                     RemoteEndpointType.S3_PRESIGNED_URL
                 ]
             )

        - Provide known file size to skip HEAD request:

          .. code-block::

             handle = RemoteFile.open(
                 "https://example.com/large-file.bin",
                 remote_endpoint_type=RemoteEndpointType.HTTP,
                 nbytes=1024 * 1024 * 100  # 100 MB
             )
        """
        return RemoteFile(
            _get_remote_module().RemoteFile.open(
                url,
                RemoteEndpointType._map_to_internal(remote_endpoint_type),
                allow_list,
                nbytes,
            )
        )

    def close(self) -> None:
        """Close the file"""
        pass

    def __enter__(self) -> RemoteFile:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def __str__(self) -> str:
        return str(self._handle)

    def remote_endpoint_type(self) -> RemoteEndpointType:
        """Get the type of the remote file.

        Returns
        -------
        The type of the remote file.
        """
        return RemoteEndpointType[self._handle.remote_endpoint_type().name]

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
