# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.


import functools
import multiprocessing as mp
import threading
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import pytest
from RangeHTTPServer import RangeRequestHandler

import kvikio
import kvikio.defaults

pytestmark = pytest.mark.skipif(
    not kvikio.is_remote_file_available(),
    reason="cannot test remote IO, please build KvikIO with libcurl",
)


def start_http_server(queue: mp.Queue, tmpdir: str, range_support: bool = True):
    handler = RangeRequestHandler if range_support else SimpleHTTPRequestHandler
    httpd = ThreadingHTTPServer(
        ("127.0.0.1", 0), functools.partial(handler, directory=tmpdir)
    )
    thread = threading.Thread(target=httpd.serve_forever)
    thread.start()
    queue.put(httpd.server_address)
    time.sleep(60)
    print("ThreadingHTTPServer shutting down because of timeout (60sec)")


@pytest.fixture
def http_server(tmpdir):
    """Fixture to set up http server in separate process"""
    queue = mp.Queue()
    p = mp.Process(target=start_http_server, args=(queue, str(tmpdir)))
    p.start()
    ip, port = queue.get()
    yield f"http://{ip}:{port}"
    p.kill()


def test_file_size(http_server, tmpdir):
    a = np.arange(100)
    a.tofile(tmpdir / "a")
    with kvikio.RemoteFile(f"{http_server}/a") as f:
        assert f.nbytes() == a.nbytes


@pytest.mark.parametrize("size", [10, 100, 1000])
@pytest.mark.parametrize("nthreads", [1, 3])
@pytest.mark.parametrize("tasksize", [99, 999])
def test_read(http_server, tmpdir, xp, size, nthreads, tasksize):
    a = xp.arange(size)
    a.tofile(tmpdir / "a")

    with kvikio.defaults.set_num_threads(nthreads):
        with kvikio.defaults.set_task_size(tasksize):
            with kvikio.RemoteFile(f"{http_server}/a") as f:
                assert f.nbytes() == a.nbytes
                b = xp.empty_like(a)
                assert f.read(buf=b) == a.nbytes
                xp.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("nthreads", [1, 10])
def test_large_read(http_server, tmpdir, xp, nthreads):
    a = xp.arange(16_000_000)
    a.tofile(tmpdir / "a")

    with kvikio.defaults.set_num_threads(nthreads):
        with kvikio.RemoteFile(f"{http_server}/a") as f:
            assert f.nbytes() == a.nbytes
            b = xp.empty_like(a)
            assert f.read(buf=b) == a.nbytes
            xp.testing.assert_array_equal(a, b)
