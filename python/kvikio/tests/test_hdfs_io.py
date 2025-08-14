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


remote_file_data = RemoteFileData("/home/test_user/test_file.bin", 1024 * 1024)


def webhdfs_server_handle_request(request: Request) -> Response:
    # request.args is a dictionary of parsed query strings
    # If client requests the file size
    if request.args["op"] == "GETFILESTATUS":
        return Response(
            response=json.dumps({"length": remote_file_data.file_size}),
            status=200,
            content_type="application/json",
        )
    if request.args["op"] == "OPEN":
        begin_idx = int(request.args["offset"])
        end_idx = begin_idx + int(request.args["length"])
        range_data = remote_file_data.buf[begin_idx:end_idx]
        return Response(
            response=range_data, status=200, content_type="application/octet-stream"
        )


@pytest.fixture
def mock_webhdfs_server(httpserver: HTTPServer) -> HTTPServer:
    # Regex meaning:
    # \? Matches the literal question mark
    # [^?]+ Matches the character that is not a question mark one or more times
    # ()? Matches the query zero or one time
    url_pattern = rf"/webhdfs/v1{remote_file_data.file_path}(\?[^?]+)?"
    httpserver.expect_request(
        re.compile(url_pattern),
    ).respond_with_handler(webhdfs_server_handle_request)

    return httpserver


def helper_get_file_size(url):
    handle = remote_file.RemoteFile.open_webhdfs(url)
    return handle.nbytes()


def test_webhdfs_get_file_size(mock_webhdfs_server: HTTPServer):
    # Given the file path, url_for prepends the scheme, host and port
    basic_url = mock_webhdfs_server.url_for(f"/webhdfs/v1{remote_file_data.file_path}")

    url_list = [basic_url, f"{basic_url}?op=OPEN"]

    with ProcessPoolExecutor() as executor:
        for url in url_list:
            fut = executor.submit(helper_get_file_size, url)
            file_size = fut.result()
            assert file_size == remote_file_data.file_size


def helper_read_parallel(url, num_elements):
    handle = remote_file.RemoteFile.open_webhdfs(url)
    result_buf = np.arange(0, num_elements, dtype=np.float64)
    fut = handle.pread(result_buf)
    read_size = fut.get()
    return read_size, result_buf


def test_webhdfs_read_parallel(mock_webhdfs_server: HTTPServer):
    url = mock_webhdfs_server.url_for(f"/webhdfs/v1{remote_file_data.file_path}")

    with ProcessPoolExecutor() as executor:
        fut = executor.submit(helper_read_parallel, url, remote_file_data.num_elements)
        read_size, result_buf = fut.result()
        assert read_size == remote_file_data.file_size
        assert np.array_equal(result_buf, remote_file_data.buf)


@pytest.fixture
def mock_bad_server(httpserver: HTTPServer) -> HTTPServer:
    httpserver.expect_request(
        f"/webhdfs/v1{remote_file_data.file_path}",
        method="GET",
        query_string={"op": "GETFILESTATUS"},
    ).respond_with_json(
        response_json={}, status=200  # Missing "length"
    )

    return httpserver


def test_webhdfs_missing_file_size(mock_bad_server: HTTPServer):
    url = mock_bad_server.url_for(f"/webhdfs/v1{remote_file_data.file_path}")

    with pytest.raises(
        RuntimeError,
        match="Regular expression search failed. "
        "Cannot extract file length from the JSON response.",
    ):
        with ProcessPoolExecutor() as executor:
            fut = executor.submit(helper_get_file_size, url)
            fut.result()
