# Copyright (c) 2021-2024, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.

import os
from importlib import import_module
from pathlib import Path

import pytest

import kvikio

examples_path = Path(os.path.realpath(__file__)).parent / ".." / "examples"


def test_hello_world(tmp_path, monkeypatch):
    """Test examples/hello_world.py"""
    pytest.importorskip("cupy")  # `examples/hello_world.py` requires CuPy

    monkeypatch.syspath_prepend(str(examples_path))
    import_module("hello_world").main(tmp_path / "test-file")


def test_zarr_cupy_nvcomp(tmp_path, monkeypatch):
    """Test examples/zarr_cupy_nvcomp.py"""

    # `examples/zarr_cupy_nvcomp.py` requires the Zarr submodule
    pytest.importorskip("kvikio.zarr")

    monkeypatch.syspath_prepend(str(examples_path))
    import_module("zarr_cupy_nvcomp").main(tmp_path / "test-file")


def test_aws_s3(monkeypatch):
    """Test examples/aws_s3.py"""

    if not kvikio.is_remote_file_available():
        pytest.skip(
            "cannot test remote IO, please build KvikIO with with AWS S3 support"
        )
    # Fail early if dependencies isn't available
    import boto3  # noqa: F401
    import moto  # noqa: F401

    monkeypatch.syspath_prepend(str(examples_path))
    import_module("aws_s3").main()
