# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.

import functools
import multiprocessing
import pathlib
import threading
import time


class LocalHttpServer:
    """Local http server - slow but convenient"""

    @staticmethod
    def _server(
        queue: multiprocessing.Queue,
        root_path: str,
        range_support: bool,
        max_lifetime: int,
    ):
        from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

        if range_support:
            from RangeHTTPServer import RangeRequestHandler

            handler = RangeRequestHandler
        else:
            handler = SimpleHTTPRequestHandler
        httpd = ThreadingHTTPServer(
            ("127.0.0.1", 0), functools.partial(handler, directory=root_path)
        )
        thread = threading.Thread(target=httpd.serve_forever)
        thread.start()
        queue.put(httpd.server_address)
        time.sleep(max_lifetime)
        print(
            f"ThreadingHTTPServer shutting down because of timeout ({max_lifetime}sec)"
        )

    def __init__(
        self, root_path: str | pathlib.Path, range_support: bool, max_lifetime: int
    ) -> None:
        self.root_path = root_path
        self.range_support = range_support
        self.max_lifetime = max_lifetime

    def __enter__(self):
        queue = multiprocessing.Queue()
        self.process = multiprocessing.Process(
            target=LocalHttpServer._server,
            args=(queue, str(self.root_path), self.range_support, self.max_lifetime),
        )
        self.process.start()
        ip, port = queue.get()
        self.ip = ip
        self.port = port
        self.url = f"http://{ip}:{port}"
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.process.kill()
