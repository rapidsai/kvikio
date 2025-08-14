# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.


import json
import re
from concurrent.futures import ProcessPoolExecutor
from typing import Tuple

import numpy as np
import numpy.typing as npt
import pytest
from pytest_httpserver import HTTPServer
from werkzeug import Request, Response
from werkzeug.datastructures import MultiDict

import kvikio.defaults
from kvikio import remote_file


class RemoteFileData:
    def __init__(self, file_path: str, num_elements: int, dtype: npt.DTypeLike):
        self.file_path = file_path
        self.num_elements = num_elements
        self.dtype = dtype
        self.buf = np.arange(0, self.num_elements, dtype=self.dtype)
        self.file_size = self.buf.nbytes


@pytest.fixture(scope="module")
def remote_file_data() -> RemoteFileData:
    return RemoteFileData("/home/test_user/test_file.bin", 1024 * 1024, np.float64)


class WebHDFSHandler:
    def __init__(self, remote_file_data: RemoteFileData):
        self.remote_file_data = remote_file_data

    def handle_request(self, request: Request) -> Response:
        op = request.args["op"]

        if op == "GETFILESTATUS":
            return self._handle_get_file_size()
        elif op == "OPEN":
            return self._handle_read(request.args)
        else:
            return Response(status=400)

    def _handle_get_file_size(self) -> Response:
        return Response(
            response=json.dumps({"length": self.remote_file_data.file_size}),
            status=200,
            content_type="application/json",
        )

    def _handle_read(self, args: MultiDict) -> Response:
        byte_offset = int(args["offset"])
        byte_length = int(args["length"])

        # Convert byte offsets to element indices
        element_size = self.remote_file_data.buf.itemsize
        begin_idx = byte_offset // element_size
        end_idx = (byte_offset + byte_length) // element_size

        range_data = self.remote_file_data.buf[begin_idx:end_idx].tobytes()
        return Response(
            response=range_data,
            status=200,
            content_type="application/octet-stream",
            headers={"Content-Length": str(len(range_data))},
        )


@pytest.fixture
def mock_webhdfs_server(
    httpserver: HTTPServer, remote_file_data: RemoteFileData
) -> HTTPServer:
    # The HTTP server and the KvikIO HTTP client must run on different processes.
    # Otherwise due to GIL, the client will hang.
    handler = WebHDFSHandler(remote_file_data)

    # Regex meaning:
    # \? Matches the literal question mark
    # [^?]+ Matches the character that is not a question mark one or more times
    # ()? Matches the query zero or one time
    url_pattern = rf"/webhdfs/v1{remote_file_data.file_path}(\?[^?]+)?"
    httpserver.expect_request(
        re.compile(url_pattern),
    ).respond_with_handler(handler.handle_request)

    return httpserver


class WebHdfsOperations:
    @staticmethod
    def get_file_size(url: str) -> int:
        handle = remote_file.RemoteFile.open_webhdfs(url)
        return handle.nbytes()

    @staticmethod
    def parallel_read(
        url: str, num_elements: int, dtype: npt.DTypeLike
    ) -> Tuple[int, np.ndarray]:
        handle = remote_file.RemoteFile.open_webhdfs(url)
        result_buf = np.arange(0, num_elements, dtype=dtype)
        fut = handle.pread(result_buf)
        read_size = fut.get()
        return read_size, result_buf

    @staticmethod
    def parallel_read_partial(
        url: str,
        num_elements: int,
        dtype: npt.DTypeLike,
        size: int,
        offset: int,
        num_threads: int,
        task_size: int,
    ) -> Tuple[int, np.ndarray]:
        actual_num_elements = size // np.dtype(dtype).itemsize
        with kvikio.defaults.set({"num_threads": num_threads, "task_size": task_size}):
            handle = remote_file.RemoteFile.open_webhdfs(url)
            result_buf = np.zeros(actual_num_elements, dtype=dtype)
            fut = handle.pread(result_buf, size, offset)
            read_size = fut.get()
            return read_size, result_buf


class TestWebHdfsOperations:
    @pytest.mark.parametrize("url_query", ["", "?op=OPEN"])
    def test_get_file_size(
        self,
        mock_webhdfs_server: HTTPServer,
        remote_file_data: RemoteFileData,
        url_query: str,
    ) -> None:
        # Given the file path, url_for prepends the scheme, host and port
        base_url = mock_webhdfs_server.url_for(
            f"/webhdfs/v1{remote_file_data.file_path}"
        )
        url = f"{base_url}{url_query}"

        with ProcessPoolExecutor() as executor:
            fut = executor.submit(WebHdfsOperations.get_file_size, url)
            file_size = fut.result()
            assert file_size == remote_file_data.file_size

    def test_parallel_read(
        self, mock_webhdfs_server: HTTPServer, remote_file_data: RemoteFileData
    ) -> None:
        url = mock_webhdfs_server.url_for(f"/webhdfs/v1{remote_file_data.file_path}")

        with ProcessPoolExecutor() as executor:
            fut = executor.submit(
                WebHdfsOperations.parallel_read,
                url,
                remote_file_data.num_elements,
                remote_file_data.dtype,
            )
            read_size, result_buf = fut.result()
            assert read_size == remote_file_data.file_size
            assert np.array_equal(result_buf, remote_file_data.buf)

    @pytest.mark.parametrize("size", [80, 8 * 9999])
    @pytest.mark.parametrize("offset", [0, 800, 8000, 8 * 9999])
    @pytest.mark.parametrize("num_threads", [1, 4])
    @pytest.mark.parametrize("task_size", [1024, 4096])
    def test_parallel_read_partial(
        self,
        mock_webhdfs_server: HTTPServer,
        remote_file_data: RemoteFileData,
        size: int,
        offset: int,
        num_threads: int,
        task_size: int,
    ) -> None:
        url = mock_webhdfs_server.url_for(f"/webhdfs/v1{remote_file_data.file_path}")

        element_size = remote_file_data.buf.itemsize
        begin_idx = offset // element_size
        end_idx = (offset + size) // element_size
        expected_buf = remote_file_data.buf[begin_idx:end_idx]
        with ProcessPoolExecutor() as executor:
            fut = executor.submit(
                WebHdfsOperations.parallel_read_partial,
                url,
                remote_file_data.num_elements,
                remote_file_data.dtype,
                size,
                offset,
                num_threads,
                task_size,
            )
            read_size, result_buf = fut.result()
            assert read_size == size
            assert np.array_equal(result_buf, expected_buf)


class TestWebHdfsErrors:
    @pytest.fixture
    def mock_bad_server(
        self, httpserver: HTTPServer, remote_file_data: RemoteFileData
    ) -> HTTPServer:
        httpserver.expect_request(
            f"/webhdfs/v1{remote_file_data.file_path}",
            method="GET",
            query_string={"op": "GETFILESTATUS"},
        ).respond_with_json(
            response_json={}, status=200  # Missing "length"
        )

        return httpserver

    def test_missing_file_size(
        self, mock_bad_server: HTTPServer, remote_file_data: RemoteFileData
    ) -> None:
        url = mock_bad_server.url_for(f"/webhdfs/v1{remote_file_data.file_path}")

        with pytest.raises(
            RuntimeError,
            match="Regular expression search failed. "
            "Cannot extract file length from the JSON response.",
        ):
            with ProcessPoolExecutor() as executor:
                fut = executor.submit(WebHdfsOperations.get_file_size, url)
                fut.result()
