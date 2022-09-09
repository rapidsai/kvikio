# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.

import os
import os.path
import sys
from pathlib import Path

import pytest

benchmarks_path = Path(os.path.realpath(__file__)).parent / ".." / "benchmarks"

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
    """Test benchmarks/single-node-io.py"""

    if "zarr" in api:
        zarr = pytest.importorskip("zarr")
        if not hasattr(zarr.Array, "meta_array"):
            pytest.skip("Requires Zarr v2.13.0+ for CuPy support")

    retcode = run_cmd(
        cmd=[
            sys.executable or "python",
            "single-node-io.py",
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
