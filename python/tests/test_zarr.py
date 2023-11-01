# Copyright (c) 2021-2023, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.


import math

import numpy
import pytest

cupy = pytest.importorskip("cupy")
zarr = pytest.importorskip("zarr")
kvikio_zarr = pytest.importorskip("kvikio.zarr")
kvikio_nvcomp_codec = pytest.importorskip("kvikio.nvcomp_codec")
numcodecs = pytest.importorskip("numcodecs")

if not kvikio_zarr.supported:
    pytest.skip(
        f"requires Zarr >={kvikio_zarr.MINIMUM_ZARR_VERSION}",
        allow_module_level=True,
    )


@pytest.fixture
def store(tmp_path):
    """Fixture that creates a GDS Store"""
    return kvikio_zarr.GDSStore(tmp_path / "test-file.zarr")


def test_direct_store_access(store, xp):
    """Test accessing the GDS Store directly"""

    a = xp.arange(5, dtype="u1")
    store["a"] = a
    b = store["a"]

    # Notice, unless using getitems(), GDSStore always returns bytes
    assert isinstance(b, bytes)
    assert (xp.frombuffer(b, dtype="u1") == a).all()


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


def test_array(store, xp):
    """Test Zarr array"""

    a = xp.arange(100)
    z = zarr.array(a, chunks=10, compressor=None, store=store, meta_array=xp.empty(()))
    assert isinstance(z.meta_array, type(a))
    assert a.shape == z.shape
    assert a.dtype == z.dtype
    assert isinstance(a, type(z[:]))
    xp.testing.assert_array_equal(a, z[:])


def test_group(store, xp):
    """Test Zarr group"""

    g = zarr.open_group(store, meta_array=xp.empty(()))
    g.ones("data", shape=(10, 11), dtype=int, compressor=None)
    a = g["data"]
    assert a.shape == (10, 11)
    assert a.dtype == int
    assert isinstance(a, zarr.Array)
    assert isinstance(a.meta_array, xp.ndarray)
    assert isinstance(a[:], xp.ndarray)
    assert (a[:] == 1).all()


def test_open_array(store, xp):
    """Test Zarr's open_array()"""

    a = xp.arange(10)
    z = zarr.open_array(
        store,
        shape=a.shape,
        dtype=a.dtype,
        chunks=(10,),
        compressor=None,
        meta_array=xp.empty(()),
    )
    z[:] = a
    assert a.shape == z.shape
    assert a.dtype == z.dtype
    assert isinstance(a, type(z[:]))
    xp.testing.assert_array_equal(a, z[:])


@pytest.mark.parametrize("inline_array", [True, False])
def test_dask_read(store, xp, inline_array):
    """Test Zarr read in Dask"""

    da = pytest.importorskip("dask.array")
    a = xp.arange(100)
    z = zarr.array(a, chunks=10, compressor=None, store=store, meta_array=xp.empty(()))
    d = da.from_zarr(z, inline_array=inline_array)
    d += 1
    xp.testing.assert_array_equal(a + 1, d.compute())


def test_dask_write(store, xp):
    """Test Zarr write in Dask"""

    da = pytest.importorskip("dask.array")

    # Write dask array to disk using Zarr
    a = xp.arange(100)
    d = da.from_array(a, chunks=10)
    da.to_zarr(d, store, compressor=None, meta_array=xp.empty(()))

    # Validate the written Zarr array
    z = zarr.open_array(store)
    xp.testing.assert_array_equal(a, z[:])


@pytest.mark.parametrize("xp_read", ["numpy", "cupy"])
@pytest.mark.parametrize("xp_write", ["numpy", "cupy"])
@pytest.mark.parametrize("compressor", kvikio_zarr.nvcomp_compressors)
def test_compressor(store, xp_write, xp_read, compressor):
    xp_read = pytest.importorskip(xp_read)
    xp_write = pytest.importorskip(xp_write)

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


@pytest.mark.parametrize("algo", ["lz4", "zstd"])
def test_decompressor_config_overwrite(tmp_path, xp, algo):
    cpu_codec = numcodecs.registry.get_codec({"id": algo})
    gpu_codec = kvikio_nvcomp_codec.NvCompBatchCodec(algo)

    # Write using Zarr's default file store and the `cpu_codec` compressor
    z = zarr.open_array(tmp_path, mode="w", shape=(10,), compressor=cpu_codec)
    z[:] = range(10)
    assert z.compressor == cpu_codec

    # Open file using GDSStore and use `gpu_codec` as decompressor.
    z = zarr.open_array(
        kvikio_zarr.GDSStore(
            tmp_path,
            decompressor_config_overwrite=gpu_codec.get_config(),
        ),
        mode="r",
        meta_array=xp.empty(()),
    )
    assert z.compressor == gpu_codec
    assert isinstance(z[:], xp.ndarray)
    xp.testing.assert_array_equal(z[:], range(10))


@pytest.mark.parametrize("algo", ["lz4"])
def test_compressor_config_overwrite(tmp_path, xp, algo):
    cpu_codec = numcodecs.registry.get_codec({"id": algo})
    gpu_codec = kvikio_nvcomp_codec.NvCompBatchCodec(algo)

    # Write file using GDSStore and the `gpu_codec` compressor. In order
    # to make the file compatible with Zarr's builtin CPU decompressor,
    # we set `cpu_codec` as the compressor in the meta file on disk.
    z = zarr.open_array(
        kvikio_zarr.GDSStore(
            tmp_path,
            compressor_config_overwrite=cpu_codec.get_config(),
            decompressor_config_overwrite=gpu_codec.get_config(),
        ),
        mode="w",
        shape=10,
        compressor=gpu_codec,
        meta_array=xp.empty(()),
    )
    assert z.compressor == gpu_codec
    z[:] = xp.arange(10)

    # We can now open the file using Zarr's builtin CPU decompressor
    z = zarr.open_array(tmp_path, mode="r")
    assert isinstance(z[:], numpy.ndarray)
    numpy.testing.assert_array_equal(z[:], range(10))


@pytest.mark.parametrize("write_mode", ["w", "w-", "a"])
@pytest.mark.parametrize("read_mode", ["r", "r+", "a"])
def test_open_cupy_array(tmp_path, write_mode, read_mode):
    a = cupy.arange(10)
    z = kvikio_zarr.open_cupy_array(
        tmp_path,
        mode=write_mode,
        shape=a.shape,
        dtype=a.dtype,
        chunks=(2,),
        compressor=kvikio_zarr.CompatCompressor.lz4(),
    )
    z[:] = a
    assert a.shape == z.shape
    assert a.dtype == z.dtype
    assert isinstance(z[:], type(a))
    assert z.compressor == kvikio_nvcomp_codec.NvCompBatchCodec("lz4")
    cupy.testing.assert_array_equal(a, z[:])

    z = kvikio_zarr.open_cupy_array(
        tmp_path,
        mode=read_mode,
    )
    assert a.shape == z.shape
    assert a.dtype == z.dtype
    assert isinstance(z[:], type(a))
    assert z.compressor == kvikio_nvcomp_codec.NvCompBatchCodec("lz4")
    cupy.testing.assert_array_equal(a, z[:])

    z = zarr.open_array(tmp_path, mode=read_mode)
    assert a.shape == z.shape
    assert a.dtype == z.dtype
    assert isinstance(z[:], numpy.ndarray)
    assert z.compressor == kvikio_zarr.CompatCompressor.lz4().cpu
    numpy.testing.assert_array_equal(a.get(), z[:])


def test_open_cupy_array_written_by_zarr(tmp_path):
    data = numpy.arange(100)
    z = zarr.open_array(
        tmp_path,
        shape=data.shape,
        mode="w",
        compressor=kvikio_zarr.CompatCompressor.lz4().cpu,
    )
    z[:] = data

    z = kvikio_zarr.open_cupy_array(tmp_path, mode="r")
    assert isinstance(z[:], cupy.ndarray)
    cupy.testing.assert_array_equal(z[:], data)


@pytest.mark.parametrize("mode", ["r", "r+", "a"])
def test_open_cupy_array_incompatible_compressor(tmp_path, mode):
    zarr.create((10,), store=tmp_path, compressor=numcodecs.Blosc())

    with pytest.raises(ValueError, match="non-CUDA compatible compressor"):
        kvikio_zarr.open_cupy_array(tmp_path, mode=mode)


def test_open_cupy_array_unknown_mode(tmp_path):
    a = cupy.arange(10)
    with pytest.raises(ValueError, match="Unknown mode: x"):
        kvikio_zarr.open_cupy_array(
            tmp_path,
            mode="x",
            shape=a.shape,
            dtype=a.dtype,
            chunks=(2,),
        )
