# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.

import os
import os.path
import subprocess
import sys
from pathlib import Path
from typing import Iterable

import pytest

benchmarks_path = Path(os.path.realpath(__file__)).parent / ".." / "benchmarks"


def run_cmd(cmd: Iterable[str], cwd=benchmarks_path, verbose=True):
    """Help function to run command"""

    res: subprocess.CompletedProcess = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=cwd
    )  # type: ignore
    if verbose:
        print(f"{cwd}$ " + " ".join(res.args))
        print(res.stdout.decode(), end="")
    return res.returncode


@pytest.mark.parametrize(
    "api",
    [
        "cufile",
        "posix",
        "cufile-mfma",
        "cufile-mf",
        "cufile-ma",
        "zarr-gds",
        "zarr-posix",
    ],
)
def test_single_node_io(tmp_path, api):
    """Test benchmarks/single-node-io.py"""

    if "zarr" in api:
        pytest.importorskip(
            "zarr.cupy",
            reason=(
                "To use Zarr arrays with GDS directly, Zarr needs CuPy support: "
                "<https://github.com/zarr-developers/zarr-python/pull/934>"
            ),
        )

    retcode = run_cmd(
        [
            sys.executable or "python",
            "single-node-io.py",
            "-n",
            "1MiB",
            "-d",
            str(tmp_path),
            "--api",
            api,
        ]
    )
    assert retcode == 0
