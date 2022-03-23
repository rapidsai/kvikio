# Copyright (c) 2021-2022, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.


import pytest

cudf = pytest.importorskip("cudf")
cupy = pytest.importorskip("cupy")
np = pytest.importorskip("numpy")
kvikio = pytest.importorskip("kvikio")
pynvcomp = pytest.importorskip("kvikio.nvcomp")


@pytest.mark.parametrize("dtype", cudf.utils.dtypes.INTEGER_TYPES)
@pytest.mark.parametrize("size", [int(1e6), int(1e7)])
def test_lz4_lib_vs_module(dtype, size):
    dtype = cupy.dtype(dtype)
    data = cupy.array(np.arange(0, (size / dtype.itemsize) - 1), dtype=dtype)
    compressor = pynvcomp.LZ4Compressor(dtype)
    compressor.compress(data)
    lib_compressor = kvikio._lib.pynvcomp._LZ4Compressor(
        pynvcomp.cp_to_nvcomp_dtype(dtype).value,
    )
    compress_temp_size = np.zeros((1,), dtype=np.int64)
    compress_out_size = np.zeros((1,), dtype=np.int64)
    lib_compressor.configure(
        data.size * data.itemsize, compress_temp_size, compress_out_size
    )
    assert compress_temp_size == compressor.compress_temp_size
    assert compress_out_size == compressor.compress_out_size


@pytest.mark.parametrize("dtype", cudf.utils.dtypes.INTEGER_TYPES)
@pytest.mark.parametrize("size", [int(1e6), int(1e7)])
def test_cascade_lib_vs_module(dtype, size):
    dtype = cupy.dtype(dtype)
    data = cupy.array(np.arange(0, (size / dtype.itemsize) - 1), dtype=dtype)
    compressor = pynvcomp.CascadedCompressor(dtype)
    compressed = compressor.compress(data)
    print("module:", compressor.compress_out_size)
    print("actual:", compressed.size)
    lib_compressor = kvikio._lib.pynvcomp._CascadedCompressor(
        pynvcomp.cp_to_nvcomp_dtype(dtype).value, 1, 1, True
    )
    compress_temp_size = np.zeros((1,), dtype=np.int64)
    compress_out_size = np.zeros((1,), dtype=np.int64)
    lib_compressor.configure(
        data.size * data.itemsize, compress_temp_size, compress_out_size
    )
    assert compressed.size < data.size * data.itemsize


@pytest.mark.parametrize("size", [int(1e5), int(1e6)])
@pytest.mark.parametrize("dtype", cudf.utils.dtypes.INTEGER_TYPES)
def test_lz4_compress(dtype, size):
    dtype = cupy.dtype(dtype)
    data = cupy.array(np.arange(0, size / dtype.itemsize), dtype=dtype)
    compressor = pynvcomp.LZ4Compressor(dtype)
    compressed = compressor.compress(data)
    decompressed = compressor.decompress(compressed)
    cupy.testing.assert_array_equal(data, decompressed)


@pytest.mark.parametrize("dtype", cudf.utils.dtypes.INTEGER_TYPES)
@pytest.mark.parametrize("size", [int(1e6), int(1e7)])
def test_cascaded_compress(dtype, size):
    dtype = cupy.dtype(dtype)
    data = cupy.array(np.arange(0, size / dtype.itemsize) - 1, dtype=dtype)
    compressor = pynvcomp.CascadedCompressor(dtype)
    compressed = compressor.compress(data)
    decompressed = compressor.decompress(compressed)
    cupy.testing.assert_array_equal(data, decompressed)
