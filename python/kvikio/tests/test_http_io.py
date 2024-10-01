# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.


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

    with kvikio.defaults.set_num_threads(nthreads):
        with kvikio.defaults.set_task_size(tasksize):
            with kvikio.RemoteFile.open_http(f"{http_server}/a") as f:
                assert f.nbytes() == a.nbytes
                b = xp.empty_like(a)
                assert f.read(b) == a.nbytes
                xp.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("nthreads", [1, 10])
def test_large_read(http_server, tmpdir, xp, nthreads):
    a = xp.arange(16_000_000)
    a.tofile(tmpdir / "a")

    with kvikio.defaults.set_num_threads(nthreads):
        with kvikio.RemoteFile.open_http(f"{http_server}/a") as f:
            assert f.nbytes() == a.nbytes
            b = xp.empty_like(a)
            assert f.read(b) == a.nbytes
            xp.testing.assert_array_equal(a, b)


def test_error_too_small_file(http_server, tmpdir, xp):
    a = xp.arange(10, dtype="uint8")
    b = xp.empty(100, dtype="uint8")
    a.tofile(tmpdir / "a")
    with kvikio.RemoteFile.open_http(f"{http_server}/a") as f:
        assert f.nbytes() == a.nbytes
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
        with pytest.raises(
            OverflowError, match="maybe the server doesn't support file ranges?"
        ):
            f.read(b, size=10, file_offset=0)
        with pytest.raises(
            OverflowError, match="maybe the server doesn't support file ranges?"
        ):
            f.read(b, size=10, file_offset=10)
