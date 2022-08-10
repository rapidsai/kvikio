# Copyright (c) 2021-2022, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.

import cupy
import numpy as np
import pytest

cudf = pytest.importorskip("cudf")
kvikio = pytest.importorskip("kvikio")
libnvcomp = pytest.importorskip("kvikio.nvcomp")


@pytest.mark.parametrize(
    "inputs",
    [
        {},
        {
            "chunk_size": 1 << 16,
            "data_type": np.uint8,
            "stream": cupy.cuda.Stream(),
            "device_id": 0,
        },
        {
            "chunk_size": 1 << 16,
        },
        {
            "data_type": np.uint8,
        },
        {
            "stream": cupy.cuda.Stream(),
        },
        {
            "device_id": 0,
        },
    ],
)
def test_lz4_compress_base(inputs):
    size = 10000
    dtype = inputs.get("data_type") if inputs.get("data_type") else np.int8
    data = cupy.array(np.arange(0, size // dtype(0).itemsize, dtype=dtype))
    compressor = libnvcomp.LZ4Compressor(**inputs)
    final = compressor.compress(data)
    assert len(final) == 401


def test_lz4_decompress_base():
    size = 10000
    dtype = cupy.int8
    data = cupy.array(np.arange(0, size // dtype(0).itemsize, dtype=dtype))
    compressor = libnvcomp.LZ4Compressor(**inputs)
    final = compressor.compress(data)
    decompressed = compressor.decompress(final)
    assert data == decompressed


@pytest.mark.parametrize("compressor", [libnvcomp.LZ4Compressor])
@pytest.mark.parametrize(
    "dtype_size",
    zip(
        [
            "uint8",
            "uint16",
            "uint32",
            "int8",
            "int16",
            "int32",
        ],
        [
            401,
            10137,
            10137,
            401,
            10137,
            10137,
        ],
    ),
)
def test_compress_dtypes(compressor, dtype_size):
    length = 10000
    expected = dtype_size[1]
    data = cupy.array(
        np.arange(
            0,
            length // cupy.dtype(dtype_size[0]).type(0).itemsize,
            dtype=dtype_size[0],
        )
    )
    compressor_instance = compressor(data_type=dtype_size[0])
    final = compressor_instance.compress(data)
    assert len(final) == expected


@pytest.mark.parametrize("dtype", cudf.utils.dtypes.INTEGER_TYPES)
@pytest.mark.parametrize("size", [int(1e6), int(1e7)])
def test_cascade_lib_vs_module(dtype, size):
    dtype = cupy.dtype(dtype)
    data = cupy.array(np.arange(0, (size / dtype.itemsize) - 1), dtype=dtype)
    compressor = libnvcomp.CascadedCompressor(dtype)
    compressed = compressor.compress(data)
    lib_compressor = kvikio._lib.libnvcomp._CascadedCompressor(
        libnvcomp.cp_to_nvcomp_dtype(dtype).value, 1, 1, True
    )
    compress_temp_size = np.zeros((1,), dtype=np.int64)
    compress_out_size = np.zeros((1,), dtype=np.int64)
    lib_compressor.configure(
        data.size * data.itemsize, compress_temp_size, compress_out_size
    )
    assert compressed.size < data.size * data.itemsize


@pytest.mark.parametrize("dtype", cudf.utils.dtypes.INTEGER_TYPES)
@pytest.mark.parametrize("size", [int(1e6), int(1e7)])
def test_cascaded_compress(dtype, size):
    dtype = cupy.dtype(dtype)
    data = cupy.array(np.arange(0, size / dtype.itemsize) - 1, dtype=dtype)
    compressor = libnvcomp.CascadedCompressor(dtype)
    compressed = compressor.compress(data)
    decompressed = compressor.decompress(compressed)
    cupy.testing.assert_array_equal(data, decompressed)
