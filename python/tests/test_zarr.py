# Copyright (c) 2021-2022, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.


import pytest

cupy = pytest.importorskip("cupy")
zarr = pytest.importorskip("zarr")
GDSStore = pytest.importorskip("kvikio.zarr").GDSStore


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


def test_array(store):
    """Test Zarr array"""

    pytest.importorskip(
        "zarr.cupy",
        reason=(
            "To use Zarr arrays with GDS directly, Zarr needs CuPy support: "
            "<https://github.com/zarr-developers/zarr-python/pull/934>"
        ),
    )

    a = cupy.arange(100)
    z = zarr.array(
        a, chunks=10, compressor=None, store=store, meta_array=cupy.empty(())
    )
    assert a.shape == z.shape
    assert a.dtype == z.dtype
    assert isinstance(a, type(z[:]))
    cupy.testing.assert_array_equal(a, z[:])


def test_group(store):
    """Test Zarr group"""

    pytest.importorskip(
        "zarr.cupy",
        reason=(
            "To use Zarr arrays with GDS directly, Zarr needs CuPy support: "
            "<https://github.com/zarr-developers/zarr-python/pull/934>"
        ),
    )

    g = zarr.open_group(store, meta_array=cupy.empty(()))
    g.ones("data", shape=(10, 11), dtype=int, compressor=None)
    a = g["data"]
    assert a.shape == (10, 11)
    assert a.dtype == int
    assert isinstance(a, zarr.Array)
    assert isinstance(a[:], cupy.ndarray)
    assert (a[:] == 1).all()
