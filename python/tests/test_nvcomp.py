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
    compressor = libnvcomp.LZ4Manager(**inputs)
    final = compressor.compress(data)
    assert len(final) == 401


def test_lz4_decompress_base():
    size = 10000
    dtype = cupy.int8
    data = cupy.array(np.arange(0, size // dtype(0).itemsize, dtype=dtype))
    compressor = libnvcomp.LZ4Manager()
    compressed = compressor.compress(data)
    decompressed = compressor.decompress(compressed)
    assert (data == decompressed).all()


@pytest.mark.parametrize(
    "compressor", [libnvcomp.LZ4Manager, libnvcomp.CascadedManager]
)
@pytest.mark.parametrize(
    "dtype",
    [
        "uint8",
        "uint16",
        "uint32",
        "int8",
        "int16",
        "int32",
    ],
)
def test_round_trip_dtypes(compressor, dtype):
    length = 10000
    data = cupy.array(
        np.arange(
            0,
            length // cupy.dtype(dtype).type(0).itemsize,
            dtype=dtype,
        )
    )
    compressor_instance = compressor(data_type=dtype)
    compressed = compressor_instance.compress(data)
    decompressed = compressor_instance.decompress(compressed)
    assert (data == decompressed).all()


@pytest.mark.parametrize(
    "inputs",
    [
        {
            "options": {
                "chunk_size": 1 << 12,
                "type": np.uint32,
                "num_RLEs": 2,
                "num_deltas": 1,
                "use_bp": True,
            },
            "stream": cupy.cuda.Stream(),
            "device_id": 0,
        },
        {
            "options": {
                "chunk_size": 1 << 12,
                "type": np.uint32,
                "num_RLEs": 2,
                "num_deltas": 1,
                "use_bp": True,
            },
            "chunk_size": 1 << 16,
        },
        {
            "options": {
                "chunk_size": 1 << 12,
                "type": np.uint32,
                "num_RLEs": 2,
                "num_deltas": 1,
                "use_bp": True,
            },
            "data_type": np.uint8,
        },
        {
            "options": {
                "chunk_size": 1 << 12,
                "type": np.uint32,
                "num_RLEs": 2,
                "num_deltas": 1,
                "use_bp": True,
            },
            "stream": cupy.cuda.Stream(),
        },
        {
            "options": {
                "chunk_size": 1 << 12,
                "type": np.uint32,
                "num_RLEs": 2,
                "num_deltas": 1,
                "use_bp": True,
            },
            "device_id": 0,
        },
    ],
)
def test_cascaded_compress_base(inputs):
    size = 10000
    dtype = inputs.get("data_type") if inputs.get("data_type") else np.int8
    data = cupy.array(np.arange(0, size // dtype(0).itemsize, dtype=dtype))
    compressor = libnvcomp.CascadedManager(**inputs)
    final = compressor.compress(data)
    assert len(final) == 624
