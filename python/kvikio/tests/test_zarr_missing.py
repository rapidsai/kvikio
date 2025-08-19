# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.

import sys

import pytest


def test_zarr_missing(monkeypatch):
    modules = list(sys.modules)
    for module in modules:
        pkg = module.split(".")[0]
        if pkg == "kvikio":
            # remove from the import cache
            monkeypatch.delitem(sys.modules, module, raising=False)
        elif pkg == "zarr":
            # force an ImportError
            monkeypatch.setitem(sys.modules, module, None)

    with pytest.raises(ImportError):
        import kvikio.zarr  # noqa: F401
