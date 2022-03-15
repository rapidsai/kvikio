# Copyright (c) 2021-2022, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.


import pytest

cudf = pytest.importorskip("cudf")
cupy = pytest.importorskip("cupy")
np = pytest.importorskip("numpy")
nvcomp = pytest.importorskip("kvikio.nvcomp")


@pytest.mark.parametrize("dtype", cudf.utils.dtypes.INTEGER_TYPES)
@pytest.mark.parametrize("size", [1, 10, int(1e6), int(1e7), int(1e8)])
def test_cascaded_compress(dtype, size):
    dtype = cupy.dtype(dtype)
    data = cupy.array(np.arange(0, size / dtype.itemsize) - 1, dtype=dtype)
    compressor = nvcomp.CascadedCompressor(dtype)
    compressed = compressor.compress(data)
    decompressed = compressor.decompress(compressed)
    cupy.testing.assert_array_equal(data, decompressed)


@pytest.mark.parametrize("size", [1, 10, int(1e6), int(1e9)])
@pytest.mark.parametrize("dtype", cudf.utils.dtypes.INTEGER_TYPES)
def test_lz4_compress(dtype, size):
    dtype = cupy.dtype(dtype)
    data = cupy.array(np.arange(0, size / dtype.itemsize), dtype=dtype)
    compressor = nvcomp.LZ4Compressor(dtype)
    compressed = compressor.compress(data)
    decompressed = compressor.decompress(compressed)
    cupy.testing.assert_array_equal(data, decompressed)
