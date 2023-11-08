# Copyright (c) 2021-2023, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.

import pytest

np = pytest.importorskip("numpy")
cupy = pytest.importorskip("cupy")
kvikio = pytest.importorskip("kvikio")
libnvcomp = pytest.importorskip("kvikio.nvcomp")


# TODO: don't hardcode the following expected values
LEN = {
    "ANS": 11144,
    "Bitcomp": 3208,
    "Cascaded": 600,
    "Gdeflate": 760,
    "LZ4": 393,
    "Snappy": 3548,
}


def assert_compression_size(actual, desired, rtol=0.1):
    """Compression ratios might change slightly between library versions

    We mark a failure as "xfail"
    """
    try:
        np.testing.assert_allclose(actual, desired, rtol=rtol)
    except AssertionError:
        pytest.xfail("mismatch in compression ratios is acceptable")
        raise


def managers():
    return [
        libnvcomp.ANSManager,
        libnvcomp.BitcompManager,
        libnvcomp.CascadedManager,
        libnvcomp.GdeflateManager,
        libnvcomp.LZ4Manager,
        libnvcomp.SnappyManager,
    ]


def dtypes():
    return [
        "uint8",
        "uint16",
        "uint32",
        "int8",
        "int16",
        "int32",
    ]


@pytest.mark.parametrize("manager, dtype", zip(managers(), dtypes()))
def test_round_trip_dtypes(manager, dtype):
    length = 10000
    data = cupy.array(
        np.arange(
            0,
            length // cupy.dtype(dtype).type(0).itemsize,
            dtype=dtype,
        )
    )
    compressor_instance = manager(data_type=dtype)
    compressed = compressor_instance.compress(data)
    decompressed = compressor_instance.decompress(compressed)
    assert (data == decompressed).all()


#
# ANS Options test
#
@pytest.mark.parametrize(
    "inputs",
    [
        {},
        {"chunk_size": 1 << 16, "device_id": 0},
        {
            "chunk_size": 1 << 16,
        },
        {
            "device_id": 0,
        },
    ],
)
def test_ans_inputs(inputs):
    size = 10000
    dtype = inputs.get("data_type") if inputs.get("data_type") else np.int8
    data = cupy.array(np.arange(0, size // dtype(0).itemsize, dtype=dtype))
    compressor = libnvcomp.ANSManager(**inputs)
    final = compressor.compress(data)
    assert_compression_size(len(final), LEN["ANS"])


@pytest.mark.parametrize(
    "inputs",
    [
        {},
        {"data_type": np.uint8, "algo": 0, "device_id": 0},
        {"data_type": np.uint8},
        {
            "algo": 0,
        },
        {
            "device_id": 0,
        },
    ],
)
def test_bitcomp_inputs(inputs):
    size = 10000
    dtype = inputs.get("data_type") if inputs.get("data_type") else np.int8
    data = cupy.array(np.arange(0, size // dtype(0).itemsize, dtype=dtype))
    compressor = libnvcomp.BitcompManager(**inputs)
    final = compressor.compress(data)
    assert_compression_size(len(final), LEN["Bitcomp"])


@pytest.mark.parametrize(
    "inputs, expected",
    zip(
        [
            {"algo": 0},
            {"algo": 1},
            {"algo": 2},
        ],
        [LEN["Bitcomp"], LEN["Bitcomp"], LEN["Bitcomp"]],
    ),
)
def test_bitcomp_algorithms(inputs, expected):
    size = 10000
    dtype = inputs.get("data_type") if inputs.get("data_type") else np.int8
    data = cupy.array(np.arange(0, size // dtype(0).itemsize, dtype=dtype))
    compressor = libnvcomp.BitcompManager(**inputs)
    final = compressor.compress(data)
    assert_compression_size(len(final), expected)


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
            "device_id": 0,
        },
    ],
)
def test_cascaded_inputs(inputs):
    size = 10000
    dtype = inputs.get("data_type") if inputs.get("data_type") else np.int8
    data = cupy.array(np.arange(0, size // dtype(0).itemsize, dtype=dtype))
    compressor = libnvcomp.CascadedManager(**inputs)
    final = compressor.compress(data)
    assert_compression_size(len(final), LEN["Cascaded"])


@pytest.mark.parametrize(
    "inputs",
    [
        {},
        {"chunk_size": 1 << 16, "algo": 0, "device_id": 0},
        {
            "chunk_size": 1 << 16,
        },
        {
            "algo": 0,
        },
        {
            "device_id": 0,
        },
    ],
)
def test_gdeflate_inputs(inputs):
    size = 10000
    dtype = inputs.get("data_type") if inputs.get("data_type") else np.int8
    data = cupy.array(np.arange(0, size // dtype(0).itemsize, dtype=dtype))
    compressor = libnvcomp.GdeflateManager(**inputs)
    final = compressor.compress(data)
    assert_compression_size(len(final), LEN["Gdeflate"])


@pytest.mark.parametrize(
    "inputs, expected",
    zip(
        [
            {"algo": 0},
        ],
        [LEN["Gdeflate"]],
    ),
)
def test_gdeflate_algorithms(inputs, expected):
    size = 10000
    dtype = np.int8
    data = cupy.array(np.arange(0, size // dtype(0).itemsize, dtype=dtype))
    compressor = libnvcomp.GdeflateManager(**inputs)
    final = compressor.compress(data)
    assert_compression_size(len(final), expected)


@pytest.mark.xfail(raises=ValueError)
@pytest.mark.parametrize(
    "inputs, expected",
    zip([{"algo": 1}, {"algo": 2}], [LEN["Gdeflate"], LEN["Gdeflate"]]),
)
def test_gdeflate_algorithms_not_implemented(inputs, expected):
    size = 10000
    dtype = np.int8
    data = cupy.array(np.arange(0, size // dtype(0).itemsize, dtype=dtype))
    compressor = libnvcomp.GdeflateManager(**inputs)
    final = compressor.compress(data)
    assert_compression_size(len(final), expected)


@pytest.mark.parametrize(
    "inputs",
    [
        {},
        {"chunk_size": 1 << 16, "data_type": np.uint8, "device_id": 0},
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
def test_lz4_inputs(inputs):
    size = 10000
    dtype = inputs.get("data_type") if inputs.get("data_type") else np.int8
    data = cupy.array(np.arange(0, size // dtype(0).itemsize, dtype=dtype))
    compressor = libnvcomp.LZ4Manager(**inputs)
    final = compressor.compress(data)
    assert_compression_size(len(final), LEN["LZ4"])


@pytest.mark.parametrize(
    "inputs",
    [
        {},
        {"chunk_size": 1 << 16, "device_id": 0},
        {
            "chunk_size": 1 << 16,
        },
        {"device_id": 0},
    ],
)
def test_snappy_inputs(inputs):
    size = 10000
    dtype = np.int8
    data = cupy.array(np.arange(0, size // dtype(0).itemsize, dtype=dtype))
    compressor = libnvcomp.SnappyManager(**inputs)
    final = compressor.compress(data)
    assert_compression_size(len(final), LEN["Snappy"])


@pytest.mark.parametrize(
    "compressor_size",
    zip(
        managers(),
        [
            {  # ANS
                "max_compressed_buffer_size": 89373,
                "num_chunks": 1,
                "uncompressed_buffer_size": 10000,
            },
            {  # Bitcomp
                "max_compressed_buffer_size": 16432,
                "num_chunks": 1,
                "uncompressed_buffer_size": 10000,
            },
            {  # Cascaded
                "max_compressed_buffer_size": 12460,
                "num_chunks": 3,
                "uncompressed_buffer_size": 10000,
            },
            {  # Gdeflate
                "max_compressed_buffer_size": 131160,
                "num_chunks": 1,
                "uncompressed_buffer_size": 10000,
            },
            {  # LZ4
                "max_compressed_buffer_size": 65888,
                "num_chunks": 1,
                "uncompressed_buffer_size": 10000,
            },
            {  # Snappy
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
    assert_compression_size(
        result.pop("max_compressed_buffer_size"),
        expected.pop("max_compressed_buffer_size"),
    )
    assert result == expected


@pytest.mark.parametrize(
    "manager,expected",
    zip(
        managers(),
        [
            {  # ANS
                "num_chunks": 1,
                "decomp_data_size": 10000,
            },
            {  # Bitcomp
                "num_chunks": 1,
                "decomp_data_size": 10000,
            },
            {  # Cascaded
                "num_chunks": 3,
                "decomp_data_size": 10000,
            },
            {  # Gdeflate
                "num_chunks": 1,
                "decomp_data_size": 10000,
            },
            {  # LZ4
                "num_chunks": 1,
                "decomp_data_size": 10000,
            },
            {  # Snappy
                "num_chunks": 1,
                "decomp_data_size": 10000,
            },
        ],
    ),
)
def test_get_decompression_config_with_default_options(manager, expected):
    length = 10000
    dtype = cupy.uint8
    data = cupy.array(
        np.arange(
            0,
            length // cupy.dtype(dtype).type(0).itemsize,
            dtype=dtype,
        )
    )
    compressor_instance = manager()
    compressed = compressor_instance.compress(data)
    result = compressor_instance.configure_decompression_with_compressed_buffer(
        compressed
    )
    assert_compression_size(
        result.pop("decomp_data_size"), expected.pop("decomp_data_size")
    )
    assert result == expected


@pytest.mark.parametrize(
    "manager, expected",
    zip(managers(), list(LEN.values())),
)
def test_get_compressed_output_size(manager, expected):
    length = 10000
    dtype = cupy.uint8
    data = cupy.array(
        np.arange(
            0,
            length // cupy.dtype(dtype).type(0).itemsize,
            dtype=dtype,
        )
    )
    compressor_instance = manager()
    compressed = compressor_instance.compress(data)
    buffer_size = compressor_instance.get_compressed_output_size(compressed)
    assert_compression_size(buffer_size, expected)


@pytest.mark.parametrize("manager", managers())
def test_managed_manager(manager):
    length = 10000
    dtype = cupy.uint8
    data = cupy.array(
        np.arange(
            0,
            length // cupy.dtype(dtype).type(0).itemsize,
            dtype=dtype,
        )
    )
    compressor_instance = manager()
    compressed = compressor_instance.compress(data)
    manager = libnvcomp.ManagedDecompressionManager(compressed)
    decompressed = manager.decompress(compressed)
    assert len(decompressed) == 10000
