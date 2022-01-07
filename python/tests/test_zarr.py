# Copyright (c) 2021-2022, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.


import pytest

cupy = pytest.importorskip("cupy")
zarr = pytest.importorskip("zarr")
GDSStore = pytest.importorskip("cufile.zarr").GDSStore


@pytest.fixture
def store(tmp_path):
    """Fixture that creates a GDS Store"""
    return GDSStore(tmp_path / "test-file.zarr")


@pytest.mark.parametrize("array_type", ["numpy", "cupy"])
def test_direct_store_access(store, array_type):
    """Test accessing the GDS Store directly"""

    module = pytest.importorskip(array_type)
    a = module.arange(5, dtype="u1")
    store["a"] = a
    b = store["a"]

    # Notice, GDSStore always returns a cupy array
    assert type(b) is cupy.ndarray
    cupy.testing.assert_array_equal(a, b)
