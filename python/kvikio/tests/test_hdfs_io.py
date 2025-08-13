# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.


import re
from concurrent.futures import ProcessPoolExecutor

import cupy as cp
import pytest
from pytest_httpserver import HTTPServer

from kvikio import remote_file


class RemoteFileData:
    def __init__(self, file_path, num_elements):
        self.file_path = file_path
        self.num_elements = num_elements
        self.buf = cp.arange(0, self.num_elements, dtype=cp.float64)
        self.file_size = self.buf.nbytes


def helper_get_file_size(url):
    handle = remote_file.RemoteFile.open_webhdfs(url)
    return handle.nbytes()


@pytest.fixture
def remote_file_data() -> RemoteFileData:
    remote_file_data = RemoteFileData("/home/test_user/test_file.bin", 1024 * 1024)
    return remote_file_data


@pytest.fixture
def good_server_for_file_size(
    httpserver: HTTPServer, remote_file_data: RemoteFileData
) -> HTTPServer:
    # Regex meaning:
    # \? Matches the literal question mark
    # [^?]+ Matches the character that is not a question mark one or more times
    # ()? Matches the query zero or one time
    url_pattern = rf"/webhdfs/v1{remote_file_data.file_path}(\?[^?]+)?"
    httpserver.expect_request(
        re.compile(url_pattern),
        method="GET",
        query_string={"op": "GETFILESTATUS"},
    ).respond_with_json(
        response_json={"length": remote_file_data.file_size}, status=200
    )

    return httpserver


def test_webhdfs_get_file_size(
    good_server_for_file_size: HTTPServer, remote_file_data: RemoteFileData
):
    # Given the file path, url_for prepends the scheme, host and port
    basic_url = good_server_for_file_size.url_for(
        f"/webhdfs/v1{remote_file_data.file_path}"
    )

    url_list = [basic_url, f"{basic_url}?op=OPEN"]

    with ProcessPoolExecutor() as executor:
        for url in url_list:
            fut = executor.submit(helper_get_file_size, url)
            file_size = fut.result()
            assert file_size == remote_file_data.file_size


@pytest.fixture
def bad_server_for_file_size(
    httpserver: HTTPServer, remote_file_data: RemoteFileData
) -> HTTPServer:
    httpserver.expect_request(
        f"/webhdfs/v1{remote_file_data.file_path}",
        method="GET",
        query_string={"op": "GETFILESTATUS"},
    ).respond_with_json(
        response_json={}, status=200  # Missing "length"
    )

    return httpserver


def test_webhdfs_bad_server(
    bad_server_for_file_size: HTTPServer, remote_file_data: RemoteFileData
):
    url = bad_server_for_file_size.url_for(f"/webhdfs/v1{remote_file_data.file_path}")

    with pytest.raises(
        RuntimeError,
        match="Regular expression search failed. "
        "Cannot extract file length from the JSON response.",
    ):
        with ProcessPoolExecutor() as executor:
            fut = executor.submit(helper_get_file_size, url)
            fut.result()


# @pytest.fixture
# def server_for_read(httpserver: HTTPServer,
#                     remote_file_data: RemoteFileData) -> HTTPServer:
#     # Optional HEAD some clients issue
#     # httpserver.expect_request(
#     #     "/webhdfs/v1/home/user/test.bin",
#     #     method="HEAD",
#     #     query_string={"op": "OPEN", "user.name": "testuser"},
#     # ).respond_with_data(b"", status=200)

#     # # Coordinator GET: send 307 redirect with Location
#     # redirect_target_path = "/webhdfs/v1/home/user/test.bin"
#     # redirect_qs = "op=OPEN&user.name=testuser&datanode=true"
#     # redirect_url = httpserver.url_for(redirect_target_path) + "?" + redirect_qs

#     # httpserver.expect_request(
#     #     redirect_target_path,
#     #     method="GET",
#     #     query_string={"op": "OPEN", "user.name": "testuser"},
#     # ).respond_with_data(
#     #     b"", status=307, headers={"Location": redirect_url}
#     # )

#     # Regex meaning:
#     # \? Matches the literal question mark
#     # [^?]+ Matches the character that is not a question mark one or more times
#     # ()? Matches the query zero or one time
#     url_pattern = fr"/webhdfs/v1{remote_file_data.file_path}(\?[^?]+)?"
#     httpserver.expect_request(re.compile(url_pattern),
#                               method="GET",
#                               query_string={"op": "GETFILESTATUS"},
#                               ).respond_with_json(
#         response_json={"length": remote_file_data.file_size},
#         status=200)

#     return httpserver


# def helper_read_parallel(url, num_elements):
#     handle = remote_file.RemoteFile.open_webhdfs(url)
#     buf = cp.arange(0, num_elements, dtype=cp.float64)
#     fut = handle.pread(buf)
#     read_size = fut.get()
#     return read_size


# def test_webhdfs_read_parallel(server_for_read: HTTPServer,
#                                remote_file_data: RemoteFileData):
#     url = server_for_read.url_for(f"/webhdfs/v1{remote_file_data.file_path}")

#     with ProcessPoolExecutor() as executor:
#         fut = executor.submit(helper_read_parallel, url,
#                               remote_file_data.num_elements)
#         read_size = fut.result()
#         assert read_size == remote_file_data.file_size
