# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import urllib.parse
from http.server import BaseHTTPRequestHandler, HTTPServer
from multiprocessing import Process, Queue
from typing import Any, Generator

import cupy as cp
import numpy as np
import numpy.typing as npt
import pytest
import utils

import kvikio.defaults
from kvikio import remote_file


class RemoteFileData:
    def __init__(self, file_path: str, num_elements: int, dtype: npt.DTypeLike) -> None:
        self.file_path = file_path
        self.num_elements = num_elements
        self.dtype = dtype
        self.buf = np.arange(0, self.num_elements, dtype=self.dtype)
        self.file_size = self.buf.nbytes


@pytest.fixture(scope="module")
def remote_file_data() -> RemoteFileData:
    return RemoteFileData(
        file_path="/webhdfs/v1/home/test_user/test_file.bin",
        num_elements=1024 * 1024,
        dtype=np.float64,
    )


def run_mock_server(queue: Queue[int], file_size: int, buf: npt.NDArray[Any]) -> None:
    """Run HTTP server in a separate process"""

    class WebHdfsHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            parsed_url = urllib.parse.urlparse(self.path)
            query_dict = urllib.parse.parse_qs(parsed_url.query)
            op = query_dict["op"]

            # Client requests file size
            if op == ["GETFILESTATUS"]:
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                response = json.dumps({"length": file_size})
                self.wfile.write(response.encode())

            # Client requests file content
            elif op == ["OPEN"]:
                offset = int(query_dict["offset"][0])
                length = int(query_dict["length"][0])

                # Convert byte offsets to element indices
                element_size = buf.itemsize
                begin_idx = offset // element_size
                end_idx = (offset + length) // element_size
                range_data = buf[begin_idx:end_idx].tobytes()

                self.send_response(200)
                self.send_header("Content-Type", "application/octet-stream")
                self.send_header("Content-Length", str(len(range_data)))
                self.end_headers()
                self.wfile.write(range_data)
            else:
                self.send_response(400)
                self.end_headers()

        def log_message(self, format: str, *args: Any) -> None:
            pass

    port = utils.find_free_port()
    server = HTTPServer((utils.localhost(), port), WebHdfsHandler)

    # Send port back to parent process
    queue.put(port)

    server.serve_forever()


@pytest.fixture
def mock_webhdfs_server(remote_file_data: RemoteFileData) -> Generator[str, None, None]:
    """Start WebHDFS mock server in a separate process"""
    queue: Queue[int] = Queue()
    server_process = Process(
        target=run_mock_server,
        args=(
            queue,
            remote_file_data.file_size,
            remote_file_data.buf,
        ),
        daemon=True,
    )
    server_process.start()

    # Get the port the server is running on
    port = queue.get(timeout=5)

    yield f"http://{utils.localhost()}:{port}"

    # Cleanup
    server_process.terminate()
    server_process.join(timeout=1)


class TestWebHdfsOperations:
    @pytest.mark.parametrize("url_query", ["", "?op=OPEN"])
    def test_get_file_size(
        self,
        mock_webhdfs_server: str,
        remote_file_data: RemoteFileData,
        url_query: str,
    ) -> None:
        url = f"{mock_webhdfs_server}{remote_file_data.file_path}{url_query}"
        handle = remote_file.RemoteFile.open_webhdfs(url)
        file_size = handle.nbytes()
        assert file_size == remote_file_data.file_size

    def test_parallel_read(
        self, mock_webhdfs_server: str, remote_file_data: RemoteFileData, xp: Any
    ) -> None:
        url = f"{mock_webhdfs_server}{remote_file_data.file_path}"
        handle = remote_file.RemoteFile.open_webhdfs(url)
        result_buf = xp.arange(
            0, remote_file_data.num_elements, dtype=remote_file_data.dtype
        )
        fut = handle.pread(result_buf)
        read_size = fut.get()

        assert read_size == remote_file_data.file_size

        result_buf_np = result_buf
        if isinstance(result_buf, cp.ndarray):
            result_buf_np = cp.asnumpy(result_buf)
        assert np.array_equal(result_buf_np, remote_file_data.buf)

    @pytest.mark.parametrize("size", [80, 8 * 9999])
    @pytest.mark.parametrize("offset", [0, 800, 8000, 8 * 9999])
    @pytest.mark.parametrize("num_threads", [1, 4])
    @pytest.mark.parametrize("task_size", [1024, 4096])
    def test_parallel_read_partial(
        self,
        mock_webhdfs_server: str,
        remote_file_data: RemoteFileData,
        size: int,
        offset: int,
        num_threads: int,
        task_size: int,
        xp: Any,
    ) -> None:
        url = f"{mock_webhdfs_server}{remote_file_data.file_path}"
        element_size = remote_file_data.buf.itemsize
        begin_idx = offset // element_size
        end_idx = (offset + size) // element_size
        expected_buf = remote_file_data.buf[begin_idx:end_idx]

        actual_num_elements = size // np.dtype(remote_file_data.dtype).itemsize
        with kvikio.defaults.set({"num_threads": num_threads, "task_size": task_size}):
            handle = remote_file.RemoteFile.open_webhdfs(url)
            result_buf = xp.zeros(actual_num_elements, dtype=remote_file_data.dtype)
            fut = handle.pread(result_buf, size, offset)
            read_size = fut.get()

            assert read_size == size

            result_buf_np = result_buf
            if isinstance(result_buf, cp.ndarray):
                result_buf_np = cp.asnumpy(result_buf)
            assert np.array_equal(result_buf_np, expected_buf)


class TestWebHdfsErrors:
    @pytest.fixture
    def mock_bad_server(
        self, remote_file_data: RemoteFileData
    ) -> Generator[str, None, None]:
        """Start a bad WebHDFS server that returns invalid JSON"""

        def run_bad_server(queue: Queue[int]) -> None:
            class BadHandler(BaseHTTPRequestHandler):
                def do_GET(self):
                    parsed = urllib.parse.urlparse(self.path)
                    query = urllib.parse.parse_qs(parsed.query)

                    if query.get("op") == ["GETFILESTATUS"]:
                        self.send_response(200)
                        self.send_header("Content-Type", "application/json")
                        self.end_headers()
                        # Missing "length" field
                        response = json.dumps({})
                        self.wfile.write(response.encode())
                    else:
                        self.send_response(400)
                        self.end_headers()

                def log_message(self, format, *args):
                    pass

            port = utils.find_free_port()
            server = HTTPServer((utils.localhost(), port), BadHandler)
            queue.put(port)
            server.serve_forever()

        queue: Queue[int] = Queue()
        server_process = Process(target=run_bad_server, args=(queue,), daemon=True)
        server_process.start()

        port = queue.get(timeout=5)

        yield f"http://{utils.localhost()}:{port}"

        server_process.terminate()
        server_process.join(timeout=1)

    def test_missing_file_size(
        self, mock_bad_server: str, remote_file_data: RemoteFileData
    ) -> None:
        url = f"{mock_bad_server}{remote_file_data.file_path}"

        with pytest.raises(
            RuntimeError,
            match="Regular expression search failed. "
            "Cannot extract file length from the JSON response.",
        ):
            handle = remote_file.RemoteFile.open_webhdfs(url)
            handle.nbytes()
