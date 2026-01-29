# SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

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


def test_http_io(tmp_path, monkeypatch):
    """Test examples/http_io.py"""

    if not kvikio.is_remote_file_available():
        pytest.skip(
            "RemoteFile not available, please build KvikIO "
            "with libcurl (-DKvikIO_REMOTE_SUPPORT=ON)"
        )

    monkeypatch.syspath_prepend(str(examples_path))
    import_module("http_io").main(tmp_path)
