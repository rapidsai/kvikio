# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.

import functools
import multiprocessing
import pathlib
import threading
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer


class LocalHttpServer:
    """Local http server - slow but convenient"""

    @staticmethod
    def _server(
        queue: multiprocessing.Queue,
        root_path: str,
        range_support: bool,
        max_lifetime: int,
    ):
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
        self,
        root_path: str | pathlib.Path,
        range_support: bool = True,
        max_lifetime: int = 120,
    ) -> None:
        """Create a context that starts a local http server.

        Example
        -------
        >>> with LocalHttpServer(root_path="/my/server/") as server:
        ...     with kvikio.RemoteFile.open_http(f"{server.url}/myfile") as f:
        ...         f.read(...)

        Parameters
        ----------
        root_path
            Path to the directory the server will serve.
        range_support
            Whether to support the ranges, required by `RemoteFile.open_http()`.
            Depend on the `RangeHTTPServer` module (`pip install rangehttpserver`).
        max_lifetime
            Maximum lifetime of the server (in seconds).
        """
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
