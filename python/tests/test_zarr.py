# Copyright (c) 2021-2023, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.


import math

import pytest

cupy = pytest.importorskip("cupy")
zarr = pytest.importorskip("zarr")
kvikio_zarr = pytest.importorskip("kvikio.zarr")

# To support CuPy arrays, we need the `meta_array` argument introduced in
# Zarr v2.13, see <https://github.com/zarr-developers/zarr-python/pull/934>
if not hasattr(zarr.Array, "meta_array"):
    pytest.skip("requires Zarr v2.13+", allow_module_level=True)


@pytest.fixture
def store(tmp_path):
    """Fixture that creates a GDS Store"""
    return kvikio_zarr.GDSStore(tmp_path / "test-file.zarr")


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

    g = zarr.open_group(store, meta_array=cupy.empty(()))
    g.ones("data", shape=(10, 11), dtype=int, compressor=None)
    a = g["data"]
    assert a.shape == (10, 11)
    assert a.dtype == int
    assert isinstance(a, zarr.Array)
    assert isinstance(a[:], cupy.ndarray)
    assert (a[:] == 1).all()


@pytest.mark.parametrize("xp_read", ["numpy", "cupy"])
@pytest.mark.parametrize("xp_write", ["numpy", "cupy"])
@pytest.mark.parametrize(
    "compressor", ["ANS", "Bitcomp", "Cascaded", "Gdeflate", "LZ4", "Snappy"]
)
def test_compressor(store, xp_write, xp_read, compressor):
    xp_read = pytest.importorskip(xp_read)
    xp_write = pytest.importorskip(xp_write)
    compressor = getattr(kvikio_zarr, compressor)

    shape = (10, 1)
    chunks = (10, 1)
    a = xp_write.arange(math.prod(shape)).reshape(shape)
    z = zarr.creation.create(
        shape=shape,
        chunks=chunks,
        compressor=compressor(),
        store=store,
        meta_array=xp_read.empty(()),
    )
    z[:] = a
    b = z[:]
    assert isinstance(b, xp_read.ndarray)
    cupy.testing.assert_array_equal(b, a)
