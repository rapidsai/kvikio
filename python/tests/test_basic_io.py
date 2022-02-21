# Copyright (c) 2021-2022, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.

import os
import random

import pytest

import kvikio
import kvikio.thread_pool

cupy = pytest.importorskip("cupy")


@pytest.mark.parametrize("size", [1, 10, 100, 1000, 1024, 4096, 4096 * 10])
@pytest.mark.parametrize("nthreads", [1, 3, 4, 16])
def test_read_write(tmp_path, size, nthreads):
    """Test basic read/write"""
    filename = tmp_path / "test-file"

    # Set number of threads KvikIO should use
    kvikio.thread_pool.reset_num_threads(nthreads)
    assert kvikio.thread_pool.get_num_threads() == nthreads

    # Write file
    a = cupy.arange(size)
    f = kvikio.CuFile(filename, "w")
    assert not f.closed
    assert f.open_flags() & (os.O_WRONLY | os.O_DIRECT | os.O_CLOEXEC)
    assert f.write(a) == a.nbytes

    # Try to read file opened in write-only mode
    with pytest.raises(RuntimeError, match="unsupported file open flags"):
        f.read(a)

    # Close file
    f.close()
    assert f.closed

    # Read file into a new array and compare
    b = cupy.empty_like(a)
    f = kvikio.CuFile(filename, "r")
    assert f.open_flags() & (os.O_RDONLY | os.O_DIRECT | os.O_CLOEXEC)
    f.read(b)
    assert all(a == b)


def test_write_in_offsets(tmp_path):
    """Write to files in chunks"""
    filename = tmp_path / "test-file"

    a = cupy.arange(200)
    f = kvikio.CuFile(filename, "w")

    nchunks = 20
    chunks = []
    file_offsets = []
    order = list(range(nchunks))
    random.shuffle(order)
    for i in order:
        chunk_size = len(a) // nchunks
        offset = i * chunk_size
        chunks.append(a[offset : offset + chunk_size])
        file_offsets.append(offset * 8)

    for i in range(nchunks):
        f.write(chunks[i], file_offset=file_offsets[i])

    f.close()
    assert f.closed

    # Read file into a new array and compare
    b = cupy.empty_like(a)
    f = kvikio.CuFile(filename, "r")
    f.read(b)
    assert all(a == b)


def test_contextmanager(tmp_path):
    """Open file using contextmanager"""
    filename = tmp_path / "test-file"
    a = cupy.arange(200)
    b = cupy.empty_like(a)
    with kvikio.CuFile(filename, "w+") as f:
        assert not f.closed
        assert f.open_flags() & (os.O_WRONLY | os.O_DIRECT | os.O_CLOEXEC)
        assert f.write(a) == a.nbytes
        f.read(b)
        assert all(a == b)
    assert f.closed


@pytest.mark.skipif(
    cupy.cuda.runtime.getDeviceCount() < 2, reason="requires multiple GPUs"
)
def test_multiple_gpus(tmp_path):
    """Test IO from two different GPUs"""
    kvikio.thread_pool.reset_num_threads(1)
    with cupy.cuda.Device(0):
        a0 = cupy.arange(200)
    with cupy.cuda.Device(1):
        a1 = cupy.zeros(200, dtype=a0.dtype)

    filename = tmp_path / "test-file"
    with kvikio.CuFile(filename, "w+") as f:
        assert f.write(a0) == a0.nbytes
    with kvikio.CuFile(filename, "r") as f:
        assert f.read(a1) == a1.nbytes
    assert all(cupy.asnumpy(a0) == cupy.asnumpy(a1))
