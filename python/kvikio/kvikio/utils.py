# Copyright (c) 2024-2025, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.

import functools
import multiprocessing
import pathlib
import threading
import time
import warnings
from http.server import (
    BaseHTTPRequestHandler,
    SimpleHTTPRequestHandler,
    ThreadingHTTPServer,
)
from typing import Any, Callable


class LocalHttpServer:
    """Local http server - slow but convenient"""

    @staticmethod
    def _server(
        queue: multiprocessing.Queue,
        handler: type[BaseHTTPRequestHandler],
        handler_options: dict[str, Any],
        max_lifetime: int,
    ):
        httpd = ThreadingHTTPServer(
            ("127.0.0.1", 0), functools.partial(handler, **handler_options)
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
        handler: type[BaseHTTPRequestHandler] | None = None,
        handler_options: dict[str, Any] | None = None,
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
        self.handler = handler
        self.handler_options = handler_options or {}

    def __enter__(self):
        queue = multiprocessing.Queue()

        if self.handler is not None:
            handler = self.handler
        elif self.range_support:
            from RangeHTTPServer import RangeRequestHandler

            handler = RangeRequestHandler
        else:
            handler = SimpleHTTPRequestHandler

        handler_options = {**self.handler_options, **{"directory": self.root_path}}

        self.process = multiprocessing.Process(
            target=LocalHttpServer._server,
            args=(queue, handler, handler_options, self.max_lifetime),
        )
        self.process.start()
        ip, port = queue.get()
        self.ip = ip
        self.port = port
        self.url = f"http://{ip}:{port}"
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.process.kill()


def call_once(func: Callable) -> Callable:
    """Decorate a function such that it is only called once

    Examples:

    .. code-block:: python

       @kvikio.utils.call_once
       foo(args)

    Parameters
    ----------
    func: Callable
        The function to be decorated.

    Returns
    -------
    Callable
        A decorated function.
    """
    once_flag = True
    cached_result = None

    def wrapper(*args, **kwargs):
        nonlocal once_flag
        nonlocal cached_result
        if once_flag:
            once_flag = False
            cached_result = func(*args, **kwargs)
        return cached_result

    return wrapper


def kvikio_deprecation_notice(
    msg: str = "This function is deprecated.", *, since: str
) -> Callable:
    """Decorate a function to print the deprecation notice at runtime,
       and also add the notice to Sphinx documentation.

    Examples:

    .. code-block:: python

       @kvikio.utils.kvikio_deprecation_notice("Use bar(args) instead.")
       foo(args)

    Parameters
    ----------
    msg: str
        The deprecation notice.

    since: str
        The KvikIO version since which the function becomes deprecated.

    Returns
    -------
    Callable
        A decorated function.
    """

    def decorator(func: Callable):
        def wrapper(*args, **kwargs):
            warnings.warn(msg, category=FutureWarning, stacklevel=2)
            return func(*args, **kwargs)

        # Allow the docstring to be correctly generated for the decorated func in Sphinx
        func_doc = getattr(func, "__doc__")
        valid_docstring = "" if func_doc is None else func_doc
        wrapper.__doc__ = "{:} {:} {:}\n\n{:}".format(
            ".. deprecated::", since, msg, valid_docstring
        )

        return wrapper

    return decorator


def kvikio_deprecate_module(msg: str = "", *, since: str) -> None:
    """Mark a module as deprecated.

    Parameters
    ----------
    msg: str
        The deprecation notice.

    since: str
        The KvikIO version since which the module becomes deprecated.
    """
    full_msg = f"This module is deprecated since {since}. {msg}"
    warnings.warn(full_msg, category=FutureWarning, stacklevel=2)
