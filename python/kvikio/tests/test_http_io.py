# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import http
import time
from http.server import SimpleHTTPRequestHandler
from typing import Literal

import numpy as np
import pytest

import kvikio
import kvikio.defaults
from kvikio.utils import LocalHttpServer

pytestmark = pytest.mark.skipif(
    not kvikio.is_remote_file_available(),
    reason=(
        "RemoteFile not available, please build KvikIO "
        "with libcurl (-DKvikIO_REMOTE_SUPPORT=ON)"
    ),
)


class RequestCounter:
    # ThreadedHTTPServer creates a new handler per request.
    # This lets us share some state between requests.
    def __init__(self):
        self.error_count = 0
        self.delay_count = 0


class HTTP503Handler(SimpleHTTPRequestHandler):
    """
    An HTTP handler that initially responds with a 503 before responding normally.

    Parameters
    ----------
    request_counter : RequestCounter
        A class with a mutable values to track the number of 503 errors and delayed
        responses that have been returned.
    max_error_count : int
        The number of times to respond with a 503 before responding normally.
    delay_duration : int
        The duration, in seconds, to sleep before responding to a GET request
        when this handler has handled fewer than `max_delay_count` requests.
    max_delay_count : int
        The maximum number of requests to delay for `delay_duration`.
    """

    def __init__(
        self,
        *args,
        directory=None,
        request_counter: RequestCounter = RequestCounter(),
        max_error_count: int = 1,
        delay_duration: int = 0,
        max_delay_count: int = 1,
        **kwargs,
    ):
        self.max_error_count = max_error_count
        self.request_counter = request_counter
        self.delay_duration = delay_duration
        self.max_delay_count = max_delay_count
        super().__init__(*args, directory=directory, **kwargs)

    def _do_with_error_count(self, method: Literal["GET", "HEAD"]) -> None:
        if self.request_counter.error_count < self.max_error_count:
            self.request_counter.error_count += 1
            self.send_error(http.HTTPStatus.SERVICE_UNAVAILABLE)
            self.send_header("CurrentErrorCount", str(self.request_counter.error_count))
            self.send_header("MaxErrorCount", str(self.max_error_count))
            return None
        else:
            if method == "GET":
                if (
                    self.delay_duration > 0
                    and self.request_counter.delay_count < self.max_delay_count
                ):
                    self.request_counter.delay_count += 1
                    time.sleep(self.delay_duration)
                return super().do_GET()
            else:
                return super().do_HEAD()

    def do_GET(self) -> None:
        return self._do_with_error_count("GET")

    def do_HEAD(self) -> None:
        return self._do_with_error_count("HEAD")


@pytest.fixture
def http_server(request, tmpdir):
    """Fixture to set up http server in separate process"""
    range_support = True
    if hasattr(request, "param"):
        range_support = request.param.get("range_support", True)

    with LocalHttpServer(tmpdir, range_support, max_lifetime=60) as server:
        yield server.url


def test_file_size(http_server, tmpdir):
    a = np.arange(100)
    a.tofile(tmpdir / "a")
    with kvikio.RemoteFile.open_http(f"{http_server}/a") as f:
        assert f.nbytes() == a.nbytes


@pytest.mark.parametrize("size", [10, 100, 1000])
@pytest.mark.parametrize("nthreads", [1, 3])
@pytest.mark.parametrize("tasksize", [99, 999])
def test_read(http_server, tmpdir, xp, size, nthreads, tasksize):
    a = xp.arange(size)
    a.tofile(tmpdir / "a")

    with kvikio.defaults.set({"num_threads": nthreads, "task_size": tasksize}):
        with kvikio.RemoteFile.open_http(f"{http_server}/a") as f:
            assert f.nbytes() == a.nbytes
            assert f"{http_server}/a" in str(f)
            b = xp.empty_like(a)
            assert f.read(b) == a.nbytes
            xp.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("nthreads", [1, 10])
def test_large_read(http_server, tmpdir, xp, nthreads):
    a = xp.arange(16_000_000)
    a.tofile(tmpdir / "a")

    with kvikio.defaults.set("num_threads", nthreads):
        with kvikio.RemoteFile.open_http(f"{http_server}/a") as f:
            assert f.nbytes() == a.nbytes
            assert f"{http_server}/a" in str(f)
            b = xp.empty_like(a)
            assert f.read(b) == a.nbytes
            xp.testing.assert_array_equal(a, b)


def test_error_too_small_file(http_server, tmpdir, xp):
    a = xp.arange(10, dtype="uint8")
    b = xp.empty(100, dtype="uint8")
    a.tofile(tmpdir / "a")
    with kvikio.RemoteFile.open_http(f"{http_server}/a") as f:
        assert f.nbytes() == a.nbytes
        assert f"{http_server}/a" in str(f)
        with pytest.raises(
            ValueError, match=r"cannot read 0\+100 bytes into a 10 bytes file"
        ):
            f.read(b)
        with pytest.raises(
            ValueError, match=r"cannot read 100\+5 bytes into a 10 bytes file"
        ):
            f.read(b, size=5, file_offset=100)


@pytest.mark.parametrize("http_server", [{"range_support": False}], indirect=True)
def test_no_range_support(http_server, tmpdir, xp):
    a = xp.arange(100, dtype="uint8")
    a.tofile(tmpdir / "a")
    b = xp.empty_like(a)
    with kvikio.RemoteFile.open_http(f"{http_server}/a") as f:
        assert f.nbytes() == a.nbytes
        assert f"{http_server}/a" in str(f)
        with pytest.raises(
            OverflowError, match="maybe the server doesn't support file ranges?"
        ):
            f.read(b, size=10, file_offset=0)
        with pytest.raises(
            OverflowError, match="maybe the server doesn't support file ranges?"
        ):
            f.read(b, size=10, file_offset=10)


def test_retry_http_503_ok(tmpdir, xp):
    a = xp.arange(100, dtype="uint8")
    a.tofile(tmpdir / "a")

    with LocalHttpServer(
        tmpdir,
        max_lifetime=60,
        handler=HTTP503Handler,
        handler_options={"request_counter": RequestCounter()},
    ) as server:
        http_server = server.url
        b = xp.empty_like(a)
        with kvikio.RemoteFile.open_http(f"{http_server}/a") as f:
            assert f.nbytes() == a.nbytes
            assert f"{http_server}/a" in str(f)
            f.read(b)


def test_retry_http_503_fails(tmpdir, xp, capfd):
    with LocalHttpServer(
        tmpdir,
        max_lifetime=60,
        handler=HTTP503Handler,
        handler_options={"request_counter": RequestCounter(), "max_error_count": 100},
    ) as server:
        a = xp.arange(100, dtype="uint8")
        a.tofile(tmpdir / "a")
        b = xp.empty_like(a)

        with (
            pytest.raises(RuntimeError) as m,
            kvikio.defaults.set("http_max_attempts", 2),
        ):
            with kvikio.RemoteFile.open_http(f"{server.url}/a") as f:
                f.read(b)

        assert m.match(r"KvikIO: HTTP request reached maximum number of attempts \(2\)")
        assert m.match("Got HTTP code 503")
        captured = capfd.readouterr()

        records = captured.out.strip().split("\n")
        assert len(records) == 1
        assert records[0] == (
            "KvikIO: Got HTTP code 503. Retrying after 500ms (attempt 1 of 2)."
        )


def test_no_retries_ok(tmpdir):
    a = np.arange(100, dtype="uint8")
    a.tofile(tmpdir / "a")

    with LocalHttpServer(
        tmpdir,
        max_lifetime=60,
    ) as server:
        http_server = server.url
        b = np.empty_like(a)
        with kvikio.defaults.set("http_max_attempts", 1):
            with kvikio.RemoteFile.open_http(f"{http_server}/a") as f:
                assert f.nbytes() == a.nbytes
                assert f"{http_server}/a" in str(f)
                f.read(b)


def test_retry_timeout_ok(tmpdir):
    a = np.arange(100, dtype="uint8")
    a.tofile(tmpdir / "a")
    with LocalHttpServer(
        tmpdir,
        max_lifetime=60,
        handler=HTTP503Handler,
        handler_options={
            "request_counter": RequestCounter(),
            "delay_duration": 2,
            "max_error_count": 0,
        },
    ) as server:
        http_server = server.url
        b = np.empty_like(a)
        with kvikio.defaults.set({"http_timeout": 1}):
            with kvikio.RemoteFile.open_http(f"{http_server}/a") as f:
                assert f.nbytes() == a.nbytes
                assert f"{http_server}/a" in str(f)
                f.read(b)


def test_set_http_status_code(tmpdir):
    with LocalHttpServer(
        tmpdir,
        max_lifetime=60,
        handler=HTTP503Handler,
        handler_options={"request_counter": RequestCounter()},
    ) as server:
        http_server = server.url
        with kvikio.defaults.set("http_status_codes", [429]):
            # this raises on the first 503 error, since it's not in the list.
            assert kvikio.defaults.get("http_status_codes") == [429]
            with pytest.raises(RuntimeError, match="503"):
                with kvikio.RemoteFile.open_http(f"{http_server}/a"):
                    pass


def test_timeout_raises(tmpdir, capfd):
    # This server / settings setup will make 2 requests.
    # The first request times out after 1s and is retried.
    # The second request times out after 1s and throws an exception.
    a = np.arange(100, dtype="uint8")
    a.tofile(tmpdir / "a")

    with LocalHttpServer(
        tmpdir,
        max_lifetime=60,
        handler=HTTP503Handler,
        handler_options={
            "request_counter": RequestCounter(),
            "max_error_count": 0,
            "delay_duration": 2,
            "max_delay_count": 10,
        },
    ) as server:
        http_server = server.url
        b = np.empty_like(a)
        with kvikio.defaults.set({"http_max_attempts": 2, "http_timeout": 1}):
            # TODO: this should raise a TimeoutError
            with pytest.raises(RuntimeError) as m:
                with kvikio.RemoteFile.open_http(f"{http_server}/a") as f:
                    assert f.nbytes() == a.nbytes
                    f.read(b)
            assert m.match("KvikIO: HTTP request reached maximum number of attempts")
            assert m.match("Operation timed out.")

    captured = capfd.readouterr()
    records = captured.out.strip().split("\n")
    assert len(records) == 1
    assert records[0] == "KvikIO: Timeout error. Retrying after 500ms (attempt 1 of 2)."
