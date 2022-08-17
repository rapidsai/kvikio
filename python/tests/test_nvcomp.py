# Copyright (c) 2021-2022, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.

import cupy
import numpy as np
import pytest

cudf = pytest.importorskip("cudf")
kvikio = pytest.importorskip("kvikio")
libnvcomp = pytest.importorskip("kvikio.nvcomp")


@pytest.mark.parametrize(
    "compressor",
    [
        libnvcomp.LZ4Manager,
        libnvcomp.CascadedManager,
        libnvcomp.SnappyManager,
    ],
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
        {},
        {
            "chunk_size": 1 << 16,
            "data_type": np.uint8,
            "device_id": 0
        },
        {
            "chunk_size": 1 << 16,
        },
        {
            "data_type": np.uint8,
        },
        {
            "device_id": 0,
        },
    ],
)
def test_lz4_compress_inputs(inputs):
    size = 10000
    dtype = inputs.get("data_type") if inputs.get("data_type") else np.int8
    data = cupy.array(np.arange(0, size // dtype(0).itemsize, dtype=dtype))
    compressor = libnvcomp.LZ4Manager(**inputs)
    final = compressor.compress(data)
    assert len(final) == 393


@pytest.mark.parametrize(
    "inputs",
    [
        {},
        {
            "chunk_size": 1 << 16,
            "device_id": 0
        },
        {
            "chunk_size": 1 << 16,
        },
        {
            "device_id": 0
        },
    ],
)
def test_snappy_compress_inputs(inputs):
    size = 10000
    dtype = inputs.get("data_type") if inputs.get("data_type") else np.int8
    data = cupy.array(np.arange(0, size // dtype(0).itemsize, dtype=dtype))
    compressor = libnvcomp.SnappyManager(**inputs)
    final = compressor.compress(data)
    assert len(final) == 3548


@pytest.mark.parametrize(
    "inputs",
    [
        {},
        {
            "options": {
                "chunk_size": 1 << 12,
                "type": np.uint32,
                "num_RLEs": 2,
                "num_deltas": 1,
                "use_bp": True,
            },
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
            "device_id": 0
        },
        {
            "options": {
                "chunk_size": 1 << 12,
                "type": np.uint32,
                "num_RLEs": 2,
                "num_deltas": 1,
                "use_bp": True,
            },
        },
    ],
)
def test_cascaded_compress_inputs(inputs):
    size = 10000
    dtype = inputs.get("data_type") if inputs.get("data_type") else np.int8
    data = cupy.array(np.arange(0, size // dtype(0).itemsize, dtype=dtype))
    compressor = libnvcomp.CascadedManager(**inputs)
    final = compressor.compress(data)
    assert len(final) == 600


@pytest.mark.parametrize(
    "compressor_size",
    zip(
        [
            libnvcomp.LZ4Manager,
            libnvcomp.CascadedManager,
            libnvcomp.SnappyManager,
        ],
        [
            {
                "max_compressed_buffer_size": 65888,
                "num_chunks": 1,
                "uncompressed_buffer_size": 10000,
            },
            {
                "max_compressed_buffer_size": 12460,
                "num_chunks": 3,
                "uncompressed_buffer_size": 10000,
            },
            {
                "max_compressed_buffer_size": 76575,
                "num_chunks": 1,
                "uncompressed_buffer_size": 10000,
            },
        ],
    ),
)
def test_get_compression_config_with_default_options(compressor_size):
    compressor = compressor_size[0]
    expected = compressor_size[1]
    length = 10000
    dtype = cupy.uint8
    data = cupy.array(
        np.arange(
            0,
            length // cupy.dtype(dtype).type(0).itemsize,
            dtype=dtype,
        )
    )
    compressor_instance = compressor()
    result = compressor_instance.configure_compression(len(data))
    assert result == expected


@pytest.mark.parametrize(
    "compressor,expected",
    zip(
        [
            libnvcomp.LZ4Manager,
            libnvcomp.CascadedManager,
            libnvcomp.SnappyManager,
        ],
        [
            {
                "num_chunks": 1,
                "decomp_data_size": 10000,
            },
            {
                "num_chunks": 3,
                "decomp_data_size": 10000,
            },
            {
                "num_chunks": 1,
                "decomp_data_size": 10000,
            },
        ],
    ),
)
def test_get_decompression_config_with_default_options(compressor, expected):
    length = 10000
    dtype = cupy.uint8
    data = cupy.array(
        np.arange(
            0,
            length // cupy.dtype(dtype).type(0).itemsize,
            dtype=dtype,
        )
    )
    compressor_instance = compressor()
    compressed = compressor_instance.compress(data)
    result = compressor_instance.configure_decompression_with_compressed_buffer(
        compressed
    )
    assert result == expected


@pytest.mark.parametrize(
    "compressor",
    [libnvcomp.LZ4Manager, libnvcomp.CascadedManager, libnvcomp.SnappyManager],
)
def test_set_scratch_buffer(compressor):
    length = 10000
    dtype = cupy.uint8
    data = cupy.array(
        np.arange(
            0,
            length // cupy.dtype(dtype).type(0).itemsize,
            dtype=dtype,
        )
    )
    compressor_instance = compressor()
    compressor_instance.configure_compression(len(data))
    buffer_size = compressor_instance.get_required_scratch_buffer_size()
    buffer = cupy.zeros(buffer_size, dtype="int8")
    compressor_instance.set_scratch_buffer(buffer)
    compressor_instance.compress(data)
    assert buffer[0] != 0


@pytest.mark.parametrize(
    "compressor,expected",
    zip(
        [
            libnvcomp.LZ4Manager,
            libnvcomp.CascadedManager,
            libnvcomp.SnappyManager,
        ],
        [252334080, 1641608, 67311208],
    ),
)
def test_get_required_scratch_buffer_size(compressor, expected):
    length = 10000
    dtype = cupy.uint8
    data = cupy.array(
        np.arange(
            0,
            length // cupy.dtype(dtype).type(0).itemsize,
            dtype=dtype,
        )
    )
    compressor_instance = compressor()
    compressor_instance.configure_compression(len(data))
    buffer_size = compressor_instance.get_required_scratch_buffer_size()
    assert buffer_size == expected


@pytest.mark.parametrize(
    "compressor,expected",
    zip(
        [
            libnvcomp.LZ4Manager,
            libnvcomp.CascadedManager,
            libnvcomp.SnappyManager,
        ],
        [393, 600, 3548],
    ),
)
def test_get_compressed_output_size(compressor, expected):
    length = 10000
    dtype = cupy.uint8
    data = cupy.array(
        np.arange(
            0,
            length // cupy.dtype(dtype).type(0).itemsize,
            dtype=dtype,
        )
    )
    compressor_instance = compressor()
    compressed = compressor_instance.compress(data)
    buffer_size = compressor_instance.get_compressed_output_size(compressed)
    assert buffer_size == expected


@pytest.mark.parametrize(
    "compressor",
    [
        libnvcomp.LZ4Manager,
        libnvcomp.CascadedManager,
        libnvcomp.SnappyManager,
    ],
)
def test_managed_manager(compressor):
    compressor = compressor
    length = 10000
    dtype = cupy.uint8
    data = cupy.array(
        np.arange(
            0,
            length // cupy.dtype(dtype).type(0).itemsize,
            dtype=dtype,
        )
    )
    compressor_instance = compressor()
    compressed = compressor_instance.compress(data)
    manager = libnvcomp.ManagedDecompressionManager(compressed)
    decompressed = manager.decompress(compressed)
    assert len(decompressed) == 10000


@pytest.mark.xfail(raises=NotImplementedError)
@pytest.mark.parametrize(
    "compressor",
    [
        libnvcomp.LZ4Manager,
        libnvcomp.CascadedManager,
        libnvcomp.SnappyManager,
    ],
)
@pytest.mark.parametrize(
    "inputs",
    [
        {"stream": cupy.cuda.Stream()},
    ],
)
def test_xfail_device_id_and_stream(compressor, inputs):
    compressor_instance = compressor(**inputs)
    assert compressor_instance is None
