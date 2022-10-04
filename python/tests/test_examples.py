# Copyright (c) 2021-2022, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.

import os
from importlib import import_module
from pathlib import Path

import pytest

examples_path = Path(os.path.realpath(__file__)).parent / ".." / "examples"


def test_hello_world(tmp_path, monkeypatch):
    """Test examples/hello_world.py"""
    pytest.importorskip("cupy")

    monkeypatch.syspath_prepend(str(examples_path))
    import_module("hello_world").main(tmp_path / "test-file")
