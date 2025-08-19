# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.

import sys

import pytest


def test_zarr_missing(monkeypatch):
    monkeypatch.setitem(sys.modules, "zarr", None)

    with pytest.raises(ImportError):
        import kvikio.zarr  # noqa: F401
