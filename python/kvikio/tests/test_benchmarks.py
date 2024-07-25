# Copyright (c) 2022-2024, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.

import os
import os.path
import sys
from pathlib import Path

import pytest

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
    """Test benchmarks/single-node-io.py"""

    if "zarr" in api:
        kz = pytest.importorskip("kvikio.zarr")
        if not kz.supported:
            pytest.skip(f"requires Zarr >={kz.MINIMUM_ZARR_VERSION}")

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
