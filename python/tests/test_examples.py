# Copyright (c) 2021-2022, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.

import os
from importlib import import_module
from pathlib import Path

examples_path = Path(os.path.realpath(__file__)).parent / ".." / "examples"


def test_hello_world(monkeypatch):
    """Test examples/hello_world.py"""
    monkeypatch.syspath_prepend(str(examples_path))
    import_module("hello_world").main()
