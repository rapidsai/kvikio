# Copyright (c) 2022-2024, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.

import os
import os.path
import sys
from pathlib import Path

import pytest

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
def test_single_node_io(run_cmd, tmp_path, api):
    """Test benchmarks/single_node_io.py"""

    if "zarr" in api:
        kz = pytest.importorskip("kvikio.zarr")
        if not kz.supported:
            pytest.skip(f"requires Zarr >={kz.MINIMUM_ZARR_VERSION}")

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
def test_zarr_io(run_cmd, tmp_path, api):
    """Test benchmarks/zarr_io.py"""

    kz = pytest.importorskip("kvikio.zarr")
    if not kz.supported:
        pytest.skip(f"requires Zarr >={kz.MINIMUM_ZARR_VERSION}")

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
        "cupy-kvikio",
        "numpy-kvikio",
        "cudf-kvikio",
        "cudf-fsspec",
    ],
)
def test_aws_s3_io(run_cmd, api):
    """Test benchmarks/aws_s3_io.py"""

    if not kvikio.is_remote_file_available():
        pytest.skip(
            "cannot test remote IO, please build KvikIO with with AWS S3 support",
            allow_module_level=True,
        )
    pytest.importorskip("boto3")
    pytest.importorskip("moto")
    if "cudf" in api:
        pytest.importorskip("cudf")

    retcode = run_cmd(
        cmd=[
            sys.executable or "python",
            "aws_s3_io.py",
            "--use-bundled-server",
            "-n",
            "1000",
            "-t",
            "4",
            "--api",
            api,
        ],
        cwd=benchmarks_path,
    )
    assert retcode == 0
