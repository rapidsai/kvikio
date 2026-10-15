# SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

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


def test_check_transfer_size_rejects_incomplete_transfers():
    from kvikio.benchmarks.utils import check_transfer_size

    check_transfer_size(16, 16)

    with pytest.raises(RuntimeError, match="expected 16 bytes, got 15"):
        check_transfer_size(15, 16)


@pytest.mark.parametrize(
    "api",
    [
        "cufile",
        "posix",
        "cufile-mfma",
        "cufile-mf",
        "cufile-ma",
    ],
)
@pytest.mark.timeout(30, method="thread")
def test_single_node_io(run_cmd, tmp_path, api):
    """Test benchmarks/single_node_io.py"""

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
