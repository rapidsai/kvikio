# Copyright (c) 2021-2023, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.

import os
import random
from contextlib import contextmanager

import pytest

import kvikio
import kvikio.defaults

cupy = pytest.importorskip("cupy")
numpy = pytest.importorskip("numpy")


def check_bit_flags(x: int, y: int) -> bool:
    """Check that the bits set in `y` is also set in `x`"""
    return x & y == y


@pytest.mark.parametrize("size", [1, 10, 100, 1000, 1024, 4096, 4096 * 10])
@pytest.mark.parametrize("nthreads", [1, 3, 4, 16])
@pytest.mark.parametrize("tasksize", [199, 1024])
def test_read_write(tmp_path, xp, size, nthreads, tasksize):
    """Test basic read/write"""
    filename = tmp_path / "test-file"

    with kvikio.defaults.set_num_threads(nthreads):
        with kvikio.defaults.set_task_size(tasksize):
            # Write file
            a = xp.arange(size)
            f = kvikio.CuFile(filename, "w")
            assert not f.closed
            assert check_bit_flags(f.open_flags(), os.O_WRONLY)
            assert f.write(a) == a.nbytes

            # Try to read file opened in write-only mode
            with pytest.raises(RuntimeError, match="unsupported file open flags"):
                f.read(a)

            # Close file
            f.close()
            assert f.closed

            # Read file into a new array and compare
            b = xp.empty_like(a)
            f = kvikio.CuFile(filename, "r")
            assert check_bit_flags(f.open_flags(), os.O_RDONLY)
            assert f.read(b) == b.nbytes
            assert all(a == b)


def test_file_handle_context(tmp_path):
    """Open a CuFile in a context"""
    filename = tmp_path / "test-file"
    a = cupy.arange(200)
    b = cupy.empty_like(a)
    with kvikio.CuFile(filename, "w+") as f:
        assert not f.closed
        assert check_bit_flags(f.open_flags(), os.O_RDWR)
        assert f.write(a) == a.nbytes
        assert f.read(b) == b.nbytes
        assert all(a == b)
    assert f.closed


@pytest.mark.skipif(
    kvikio.defaults.compat_mode(),
    reason="cannot test `set_compat_mode` when already running in compatibility mode",
)
def test_set_compat_mode_between_io(tmp_path):
    """Test changing `compat_mode`"""

    with kvikio.defaults.set_compat_mode(False):
        f = kvikio.CuFile(tmp_path / "test-file", "w")
        assert not f.closed
        assert f.open_flags() & os.O_WRONLY != 0
        with kvikio.defaults.set_compat_mode(True):
            a = cupy.arange(10)
            assert f.write(a) == a.nbytes


def test_write_to_files_in_chunks(tmp_path, xp):
    """Write to files in chunks"""
    filename = tmp_path / "test-file"

    a = xp.arange(200)
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
    b = xp.empty_like(a)
    f = kvikio.CuFile(filename, "r")
    assert f.read(b) == b.nbytes
    assert all(a == b)


@pytest.mark.parametrize("nthreads", [1, 3, 16])
@pytest.mark.parametrize("tasksize", [1000, 4096, int(1.5 * 4096), int(2.3 * 4096)])
@pytest.mark.parametrize(
    "start,end",
    [(0, 10 * 4096), (1, int(1.3 * 4096)), (int(2.1 * 4096), int(5.6 * 4096))],
)
def test_read_write_slices(tmp_path, xp, nthreads, tasksize, start, end):
    """Read and write different slices"""

    with kvikio.defaults.set_num_threads(nthreads):
        with kvikio.defaults.set_task_size(tasksize):
            filename = tmp_path / "test-file"
            a = xp.arange(10 * 4096)  # 10 page-sizes
            b = a.copy()
            a[start:end] = 42
            with kvikio.CuFile(filename, "w") as f:
                assert f.write(a[start:end]) == a[start:end].nbytes
            with kvikio.CuFile(filename, "r") as f:
                assert f.read(b[start:end]) == b[start:end].nbytes
            assert all(a == b)


@pytest.mark.parametrize("size", [1, 10, 100, 1000, 1024, 4096, 4096 * 10])
def test_raw_read_write(tmp_path, size):
    """Test raw read/write"""
    filename = tmp_path / "test-file"

    a = cupy.arange(size)
    with kvikio.CuFile(filename, "w") as f:
        assert f.raw_write(a) == a.nbytes
    with kvikio.CuFile(filename, "r") as f:
        assert f.raw_read(a) == a.nbytes


def test_raw_read_write_of_host_memory(tmp_path):
    """Test raw read/write of host memory, which isn't supported"""
    filename = tmp_path / "test-file"

    a = numpy.arange(1024)
    with kvikio.CuFile(filename, "w") as f:
        with pytest.raises(ValueError, match="Non-CUDA buffers not supported"):
            f.raw_write(a)
    with kvikio.CuFile(filename, "r") as f:
        with pytest.raises(ValueError, match="Non-CUDA buffers not supported"):
            assert f.raw_read(a) == a.nbytes


@contextmanager
def with_no_cuda_context():
    """Context that pop all CUDA contexts before the test and push them back on after"""
    cuda = pytest.importorskip("cuda.cuda")
    assert cuda.cuInit(0)[0] == cuda.CUresult.CUDA_SUCCESS

    ctx_stack = []
    while True:
        err, ctx = cuda.cuCtxPopCurrent()
        if err == cuda.CUresult.CUDA_ERROR_INVALID_CONTEXT:
            break
        assert err == cuda.CUresult.CUDA_SUCCESS
        ctx_stack.append(ctx)
    yield
    for ctx in reversed(ctx_stack):
        (err,) = cuda.cuCtxPushCurrent(ctx)
        assert err == cuda.CUresult.CUDA_SUCCESS


def test_no_current_cuda_context(tmp_path):
    filename = tmp_path / "test-file"
    ary = cupy.arange(100)

    with with_no_cuda_context():
        with kvikio.CuFile(filename, "w") as f:
            f.write(ary)

    with cupy.cuda.using_allocator(cupy.cuda.malloc_async):
        ary = cupy.arange(100)

    with with_no_cuda_context():
        with kvikio.CuFile(filename, "w") as f:
            f.write(ary)

    with kvikio.CuFile(filename, "w") as f:
        f.write(ary)


@pytest.mark.skipif(
    cupy.cuda.runtime.getDeviceCount() < 2, reason="requires multiple GPUs"
)
def test_multiple_gpus(tmp_path):
    """Test IO from two different GPUs"""
    with kvikio.defaults.set_num_threads(10):
        with kvikio.defaults.set_task_size(10):
            with cupy.cuda.Device(0):
                a0 = cupy.arange(200)
            with cupy.cuda.Device(1):
                a1 = cupy.zeros(200, dtype=a0.dtype)

            filename = tmp_path / "test-file"
            with kvikio.CuFile(filename, "w") as f:
                with cupy.cuda.Device(0):
                    assert f.write(a0) == a0.nbytes

            with kvikio.CuFile(filename, "r") as f:
                with pytest.raises(
                    RuntimeError,
                    match="The current CUDA context must own the given device memory",
                ):
                    f.read(a1)
                with cupy.cuda.Device(1):
                    assert f.read(a1) == a1.nbytes
            assert all(cupy.asnumpy(a0) == cupy.asnumpy(a1))
