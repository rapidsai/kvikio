# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.

import itertools as it
import json

import numcodecs
import numpy as np
import pytest
import zarr
from numpy.testing import assert_equal

from kvikio.nvcomp_codec import NvCompBatchCodec

NVCOMP_CODEC_ID = "nvcomp_batch"

LZ4_ALGO = "LZ4"
GDEFLATE_ALGO = "Gdeflate"
SNAPPY_ALGO = "snappy"
ZSTD_ALGO = "zstd"

SUPPORTED_CODECS = [LZ4_ALGO, GDEFLATE_ALGO, SNAPPY_ALGO, ZSTD_ALGO]


def _get_codec(algo: str, **kwargs):
    codec_args = {"id": NVCOMP_CODEC_ID, "algorithm": algo, "options": kwargs}
    return numcodecs.registry.get_codec(codec_args)


@pytest.fixture(params=[(32,), (8, 16), (16, 16)])
def shape(request):
    return request.param


# Separate fixture for combinations of shapes and chunks, since
# chunks array must have the same rank as data array.
@pytest.fixture(
    params=it.chain(
        it.product([(64,)], [(64,), (100,)]),
        it.product([(16, 8), (16, 16)], [(8, 16), (16, 16), (40, 12)]),
    )
)
def shape_chunks(request):
    return request.param


@pytest.mark.parametrize("algo", SUPPORTED_CODECS)
def test_codec_registry(algo: str):
    codec = _get_codec(algo)
    assert isinstance(codec, numcodecs.abc.Codec)


@pytest.mark.parametrize("algo", SUPPORTED_CODECS)
def test_basic(algo: str, shape):
    codec = NvCompBatchCodec(algo)

    # Create data.
    dtype = np.float32
    data = np.ones(shape, dtype=dtype)
    # Do roundtrip.
    comp_data = codec.encode(data)
    # Decompress and cast to original data type/shape.
    decomp_data = codec.decode(comp_data).view(dtype).reshape(shape)

    assert_equal(decomp_data, data)


@pytest.mark.parametrize("algo", SUPPORTED_CODECS)
def test_basic_zarr(algo: str, shape_chunks):
    shape, chunks = shape_chunks

    codec = NvCompBatchCodec(algo)

    data = np.ones(shape, dtype=np.float32)

    # This will do the compression.
    z = zarr.array(data, chunks=chunks, compressor=codec)

    # Test the decompression.
    assert_equal(z[:], data[:])


@pytest.mark.parametrize("algo", SUPPORTED_CODECS)
def test_batch_comp_decomp(algo: str):
    codec = _get_codec(algo)

    np.random.seed(1)

    dtype = np.float32
    # 2 equal-sized chunks.
    chunks = list(np.random.randn(2, 100).astype(dtype))

    comp_chunks = codec.encode_batch([c.tobytes() for c in chunks])
    assert len(comp_chunks) == 2

    decomp_chunks = codec.decode_batch(comp_chunks)
    assert len(decomp_chunks) == 2

    decomp_chunks = [c.view(dtype=dtype) for c in decomp_chunks]
    assert_equal(decomp_chunks[0], chunks[0])
    assert_equal(decomp_chunks[1], chunks[1])


@pytest.mark.parametrize("algo", SUPPORTED_CODECS)
def test_comp_decomp(algo: str, shape_chunks):
    shape, chunks = shape_chunks

    codec = _get_codec(algo)

    np.random.seed(1)

    data = np.random.randn(*shape).astype(np.float32)

    z1 = zarr.array(data, chunks=chunks, compressor=codec)

    zarr_store = zarr.MemoryStore()
    zarr.save_array(zarr_store, z1, compressor=codec)
    # Check the store.
    meta = json.loads(zarr_store[".zarray"])
    assert meta["compressor"]["id"] == NVCOMP_CODEC_ID
    assert meta["compressor"]["algorithm"] == algo.lower()

    # Read back/decompress.
    z2 = zarr.open_array(zarr_store)

    assert_equal(z1[:], z2[:])


@pytest.mark.parametrize(
    "algo, options",
    [
        ("lz4", {"data_type": 4}),  # NVCOMP_TYPE_INT data type.
        ("gdeflate", {"algo": 1}),  # low-throughput, high compression ratio algo
    ],
)
def test_codec_options(algo, options):
    codec = NvCompBatchCodec(algo, options)

    shape = (16, 16)
    chunks = (8, 8)

    data = np.ones(shape, dtype=np.float32)

    z = zarr.array(data, chunks=chunks, compressor=codec)

    assert_equal(z[:], data[:])


def test_codec_invalid_options():
    # There are currently only 3 supported algos in Gdeflate
    codec = NvCompBatchCodec(GDEFLATE_ALGO, options={"algo": 10})

    data = np.ones((16, 16), dtype=np.float32)

    with pytest.raises(RuntimeError):
        zarr.array(data, compressor=codec)


@pytest.mark.parametrize(
    "cpu_algo, gpu_algo",
    [
        ("lz4", LZ4_ALGO),
        ("zstd", ZSTD_ALGO),
    ],
)
def test_cpu_comp_gpu_decomp(cpu_algo, gpu_algo):
    cpu_codec = numcodecs.registry.get_codec({"id": cpu_algo})
    gpu_codec = _get_codec(gpu_algo)

    shape = (16, 16)
    chunks = (8, 8)

    data = np.ones(shape, dtype=np.float32)

    z1 = zarr.array(data, chunks=chunks)
    store = {}
    zarr.save_array(store, z1, compressor=cpu_codec)

    meta = json.loads(store[".zarray"])
    assert meta["compressor"]["id"] == cpu_algo

    meta["compressor"] = {"id": NVCOMP_CODEC_ID, "algorithm": gpu_algo}
    store[".zarray"] = json.dumps(meta).encode()

    z2 = zarr.open_array(store, compressor=gpu_codec)

    assert_equal(z1[:], z2[:])


def test_lz4_codec_header(shape_chunks):
    shape, chunks = shape_chunks

    # Test LZ4 nvCOMP codecs with and without the header.
    codec_h = _get_codec(LZ4_ALGO, has_header=True)
    codec_no_h = _get_codec(LZ4_ALGO, has_header=False)

    np.random.seed(1)

    data = np.random.randn(*shape).astype(np.float32)

    z_h = zarr.array(data, chunks=chunks, compressor=codec_h)
    z_no_h = zarr.array(data, chunks=chunks, compressor=codec_no_h)

    # Result must be the same regardless of the header presence.
    assert_equal(z_h[:], z_no_h[:])


def test_empty_batch():
    codec = _get_codec(LZ4_ALGO)

    assert len(codec.encode_batch([])) == 0
    assert len(codec.decode_batch([])) == 0
