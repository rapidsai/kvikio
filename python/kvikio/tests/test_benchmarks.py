# Copyright (c) 2022-2025, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.

import os
import os.path
import sys
from pathlib import Path

import pytest
from packaging.version import parse

import kvikio

benchmarks_path = (
    Path(os.path.realpath(__file__)).parent.parent / "kvikio" / "benchmarks"
)
pytest.importorskip("cupy")
pytest.importorskip("dask")


@pytest.mark.parametrize(
    "api",
    [
        "cufile",
        "posix",
        "cufile-mfma",
        "cufile-mf",
        "cufile-ma",
        "zarr",
    ],
)
@pytest.mark.timeout(30, method="thread")
def test_single_node_io(run_cmd, tmp_path, api):
    """Test benchmarks/single_node_io.py"""

    if "zarr" in api:
        kz = pytest.importorskip("kvikio.zarr")
        import zarr

        if not kz.supported:
            pytest.skip(f"requires Zarr >={kz.MINIMUM_ZARR_VERSION}")

        if parse(zarr.__version__) >= parse("3.0.0"):
            pytest.skip(
                "requires Zarr<3",
            )

    retcode = run_cmd(
        cmd=[
            sys.executable or "python",
            "single_node_io.py",
            "-n",
            "1MiB",
            "-d",
            str(tmp_path),
            "--api",
            api,
        ],
        cwd=benchmarks_path,
    )
    assert retcode == 0


@pytest.mark.parametrize(
    "api",
    [
        "kvikio",
        "posix",
    ],
)
@pytest.mark.timeout(30, method="thread")
def test_zarr_io(run_cmd, tmp_path, api):
    """Test benchmarks/zarr_io.py"""

    kz = pytest.importorskip("kvikio.zarr")
    import zarr

    if not kz.supported:
        pytest.skip(f"requires Zarr >={kz.MINIMUM_ZARR_VERSION}")

    if parse(zarr.__version__) >= parse("3.0.0"):
        pytest.skip(
            "requires Zarr<3",
        )

    retcode = run_cmd(
        cmd=[
            sys.executable or "python",
            "zarr_io.py",
            "-n",
            "1MiB",
            "-d",
            str(tmp_path),
            "--api",
            api,
        ],
        cwd=benchmarks_path,
    )
    assert retcode == 0


@pytest.mark.parametrize(
    "api",
    [
        "cupy",
        "numpy",
    ],
)
@pytest.mark.timeout(30, method="thread")
def test_http_io(run_cmd, api):
    """Test benchmarks/http_io.py"""

    if not kvikio.is_remote_file_available():
        pytest.skip(
            "RemoteFile not available, please build KvikIO "
            "with libcurl (-DKvikIO_REMOTE_SUPPORT=ON)"
        )
    retcode = run_cmd(
        cmd=[
            sys.executable,
            "http_io.py",
            "-n",
            "1000",
            "--api",
            api,
        ],
        cwd=benchmarks_path,
    )
    assert retcode == 0


@pytest.mark.parametrize(
    "api",
    [
        "cupy",
        "numpy",
    ],
)
@pytest.mark.timeout(30, method="thread")
def test_s3_io(run_cmd, api):
    """Test benchmarks/s3_io.py"""

    if not kvikio.is_remote_file_available():
        pytest.skip(
            "RemoteFile not available, please build KvikIO "
            "with libcurl (-DKvikIO_REMOTE_SUPPORT=ON)"
        )
    retcode = run_cmd(
        cmd=[
            sys.executable,
            "http_io.py",
            "-n",
            "1000",
            "--api",
            api,
        ],
        cwd=benchmarks_path,
    )
    assert retcode == 0
