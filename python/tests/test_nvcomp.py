# Copyright (c) 2021-2022, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.


import pytest

cupy = pytest.importorskip("cupy")
nvcomp = pytest.importorskip("kvikio.nvcomp")


# @pytest.mark.parametrize("array_type", ["numpy", "cupy"])
# @pytest.mark.parametrize(all dtypes)
# @pytest.mark.parametrize(many sizes)
def test_cascaded_compress():
    dtype = cupy.int8
    data = cupy.array(range(0, 10), dtype=dtype)
    compressor = nvcomp.CascadedCompressor(dtype)
    compressed = compressor.compress(data)
    decompressed = compressor.decompress(compressed)
    cupy.testing.assert_array_equal(data, decompressed)


def test_lz4_compress():
    dtype = cupy.int8
    data = cupy.array(range(0, 1024), dtype=dtype)
    compressor = nvcomp.LZ4Compressor(dtype)
    compressed = compressor.compress(data)
    decompressed = compressor.decompress(compressed)
    cupy.testing.assert_array_equal(data, decompressed)
