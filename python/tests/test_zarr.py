# Copyright (c) 2021-2022, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.


import pytest

cupy = pytest.importorskip("cupy")
zarr = pytest.importorskip("zarr")
GDSStore = pytest.importorskip("kvikio.zarr").GDSStore

# To support CuPy arrays, we need the `meta_array` argument introduced in
# Zarr v2.13, see <https://github.com/zarr-developers/zarr-python/pull/934>
if not hasattr(zarr.Array, "meta_array"):
    pytest.skip("requires Zarr v2.13+", allow_module_level=True)


@pytest.fixture
def store(tmp_path):
    """Fixture that creates a GDS Store"""
    cupy.arange(1)  # Making sure that CUDA has been initialized
    return GDSStore(tmp_path / "test-file.zarr")


@pytest.mark.parametrize("array_type", ["numpy", "cupy"])
def test_direct_store_access(store, array_type):
    """Test accessing the GDS Store directly"""

    xp = pytest.importorskip(array_type)
    a = xp.arange(5, dtype="u1")
    store["a"] = a
    b = store["a"]

    # Notice, when not using getitems(), GDSStore returns bytes always
    assert isinstance(b, bytes)
    a.data == b


@pytest.mark.parametrize("xp_write", ["numpy", "cupy"])
@pytest.mark.parametrize("xp_read_a", ["numpy", "cupy"])
@pytest.mark.parametrize("xp_read_b", ["numpy", "cupy"])
def test_direct_store_access_getitems(store, xp_write, xp_read_a, xp_read_b):
    """Test accessing the GDS Store directly using getitems()"""

    xp_read_a = pytest.importorskip(xp_read_a)
    xp_read_b = pytest.importorskip(xp_read_b)
    xp_write = pytest.importorskip(xp_write)
    a = xp_write.arange(5, dtype="u1")
    b = a * 2
    store["a"] = a
    store["b"] = b

    res = store.getitems(
        keys=["a", "b"],
        contexts={
            "a": {"meta_array": xp_read_a.empty(())},
            "b": {"meta_array": xp_read_b.empty(())},
        },
    )
    assert isinstance(res["a"], xp_read_a.ndarray)
    assert isinstance(res["b"], xp_read_b.ndarray)
    cupy.testing.assert_array_equal(res["a"], a)
    cupy.testing.assert_array_equal(res["b"], b)


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
