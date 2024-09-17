# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.


import functools
import multiprocessing as mp
import threading
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import pytest

import kvikio
import kvikio.defaults

pytestmark = pytest.mark.skipif(
    not kvikio.is_remote_file_available(),
    reason="cannot test remote IO, please build KvikIO with libcurl",
)


def start_http_server(queue: mp.Queue, tmpdir: str):
    httpd = ThreadingHTTPServer(
        ("127.0.0.1", 0), functools.partial(SimpleHTTPRequestHandler, directory=tmpdir)
    )
    thread = threading.Thread(target=httpd.serve_forever)
    thread.start()
    queue.put(httpd.server_address)
    time.sleep(60)
    print("ThreadingHTTPServer shutting down because of timeout (60sec)")


@pytest.fixture  # (scope="session")
def http_server(tmpdir):
    """Fixture to set up http server in separate process"""
    print(str(tmpdir))
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
