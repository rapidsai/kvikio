# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.


import json
import re
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pytest
from pytest_httpserver import HTTPServer
from werkzeug import Request, Response

from kvikio import remote_file


class RemoteFileData:
    def __init__(self, file_path, num_elements):
        self.file_path = file_path
        self.num_elements = num_elements
        self.buf = np.arange(0, self.num_elements, dtype=np.float64)
        self.file_size = self.buf.nbytes


@pytest.fixture(scope="module")
def remote_file_data():
    """Fixture providing test file data."""
    return RemoteFileData("/home/test_user/test_file.bin", 1024 * 1024)


class WebHDFSHandler:
    def __init__(self, remote_file_data: RemoteFileData):
        self.remote_file_data = remote_file_data

    def handle_request(self, request: Request) -> Response:
        """Handle WebHDFS API requests."""
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

    def _handle_read(self, args) -> Response:
        begin_idx = int(args["offset"])
        end_idx = begin_idx + int(args["length"])
        range_data = self.remote_file_data.buf[begin_idx:end_idx]
        return Response(
            response=range_data, status=200, content_type="application/octet-stream"
        )


@pytest.fixture
def mock_webhdfs_server(
    httpserver: HTTPServer, remote_file_data: RemoteFileData
) -> HTTPServer:
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
    def get_file_size(url):
        handle = remote_file.RemoteFile.open_webhdfs(url)
        return handle.nbytes()

    @staticmethod
    def read_parallel(url, num_elements):
        handle = remote_file.RemoteFile.open_webhdfs(url)
        result_buf = np.arange(0, num_elements, dtype=np.float64)
        fut = handle.pread(result_buf)
        read_size = fut.get()
        return read_size, result_buf


class TestWebHdfsOperations:
    @pytest.mark.parametrize("url_query", ["", "?op=OPEN"])
    def test_get_file_size(
        self,
        mock_webhdfs_server: HTTPServer,
        remote_file_data: RemoteFileData,
        url_query: str,
    ):
        # Given the file path, url_for prepends the scheme, host and port
        base_url = mock_webhdfs_server.url_for(
            f"/webhdfs/v1{remote_file_data.file_path}"
        )
        url = f"{base_url}{url_query}"

        with ProcessPoolExecutor() as executor:
            fut = executor.submit(WebHdfsOperations.get_file_size, url)
            file_size = fut.result()
            assert file_size == remote_file_data.file_size

    def test_read_parallel(
        self, mock_webhdfs_server: HTTPServer, remote_file_data: RemoteFileData
    ):
        url = mock_webhdfs_server.url_for(f"/webhdfs/v1{remote_file_data.file_path}")

        with ProcessPoolExecutor() as executor:
            fut = executor.submit(
                WebHdfsOperations.read_parallel, url, remote_file_data.num_elements
            )
            read_size, result_buf = fut.result()
            assert read_size == remote_file_data.file_size
            assert np.array_equal(result_buf, remote_file_data.buf)


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
    ):
        url = mock_bad_server.url_for(f"/webhdfs/v1{remote_file_data.file_path}")

        with pytest.raises(
            RuntimeError,
            match="Regular expression search failed. "
            "Cannot extract file length from the JSON response.",
        ):
            with ProcessPoolExecutor() as executor:
                fut = executor.submit(WebHdfsOperations.get_file_size, url)
                fut.result()
