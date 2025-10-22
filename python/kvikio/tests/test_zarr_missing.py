# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys

import pytest


def test_zarr_missing_raises(monkeypatch):
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


def test_zarr_2_installed_raises(monkeypatch):
    modules = list(sys.modules)
    zarr = pytest.importorskip("zarr")
    monkeypatch.setattr(zarr, "__version__", "2.0.0")

    for module in modules:
        pkg = module.split(".")[0]
        if pkg == "kvikio":
            # remove from the import cache
            monkeypatch.delitem(sys.modules, module, raising=False)

    with pytest.raises(ImportError):
        import kvikio.zarr  # noqa: F401
