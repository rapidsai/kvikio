# Copyright (c) 2021-2023, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.

import os
from importlib import import_module
from pathlib import Path

import pytest

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
