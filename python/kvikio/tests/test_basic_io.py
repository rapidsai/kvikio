# SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import io
import os
import random
from contextlib import contextmanager

import pytest

import kvikio
import kvikio.buffer
import kvikio.defaults

cupy = pytest.importorskip("cupy")
numpy = pytest.importorskip("numpy")


def check_bit_flags(x: int, y: int) -> bool:
    """Check that the bits set in `y` is also set in `x`"""
    return x & y == y


@pytest.mark.parametrize("size", [1, 10, 100, 1000, 1024, 4096, 4096 * 10])
@pytest.mark.parametrize("nthreads", [1, 3, 4, 16])
@pytest.mark.parametrize("tasksize", [199, 1024])
def test_write(tmp_path, xp, gds_threshold, size, nthreads, tasksize):
    """Test basic read/write"""
    filename = tmp_path / "test-file"

    with kvikio.defaults.set({"num_threads": nthreads, "task_size": tasksize}):
        a = xp.arange(size)
        f = kvikio.CuFile(filename, "w")
        assert not f.closed
        assert check_bit_flags(f.open_flags(), os.O_WRONLY)
        assert f.write(a) == a.nbytes
        f.close()
        assert f.closed

        b = numpy.fromfile(filename, dtype=a.dtype)
        xp.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("size", [1, 10, 100, 1000, 1024, 4096, 4096 * 10])
@pytest.mark.parametrize("nthreads", [1, 3, 4, 16])
@pytest.mark.parametrize("tasksize", [199, 1024])
def test_read(tmp_path, xp, gds_threshold, size, nthreads, tasksize):
    """Test basic read/write"""
    filename = tmp_path / "test-file"

    with kvikio.defaults.set({"num_threads": nthreads, "task_size": tasksize}):
        a = numpy.arange(size)
        a.tofile(filename)
        os.sync()

        b = xp.empty_like(a)
        f = kvikio.CuFile(filename, "r")
        assert check_bit_flags(f.open_flags(), os.O_RDONLY)
        assert f.read(b) == b.nbytes
        xp.testing.assert_array_equal(a, b)


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
        cupy.testing.assert_array_equal(a, b)
    assert f.closed


def test_no_file_error(tmp_path):
    """Test "No such file" error"""

    filename = tmp_path / "test-file"
    with pytest.raises(
        RuntimeError, match="Unable to open file.*No such file or directory"
    ):
        kvikio.CuFile(filename, "r")


def test_incorrect_open_mode_error(tmp_path, xp):
    """Test incorrect mode errors"""
    filename = tmp_path / "test-file"
    a = numpy.arange(10)
    a.tofile(filename)
    os.sync()

    with kvikio.CuFile(filename, "r") as f:
        with pytest.raises(RuntimeError, match="Operation not permitted"):
            f.write(xp.arange(10))

    with kvikio.CuFile(filename, "w") as f:
        with pytest.raises(RuntimeError, match="Operation not permitted"):
            f.read(xp.arange(10))


@pytest.mark.skipif(
    kvikio.defaults.is_compat_mode_preferred(),
    reason="cannot test `set_compat_mode` when already running in compatibility mode",
)
def test_set_compat_mode_between_io(tmp_path):
    """Test changing `compat_mode`"""
    with kvikio.defaults.set("compat_mode", kvikio.CompatMode.OFF):
        f = kvikio.CuFile(tmp_path / "test-file", "w")
        assert not f.closed
        assert f.open_flags() & os.O_WRONLY != 0
        with kvikio.defaults.set("compat_mode", kvikio.CompatMode.ON):
            a = cupy.arange(10)
            assert f.write(a) == a.nbytes


def test_write_to_files_in_chunks(tmp_path, xp, gds_threshold):
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
    xp.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("nthreads", [1, 3, 16])
@pytest.mark.parametrize("tasksize", [1000, 4096, int(1.5 * 4096), int(2.3 * 4096)])
@pytest.mark.parametrize(
    "start,end",
    [(0, 10 * 4096), (1, int(1.3 * 4096)), (int(2.1 * 4096), int(5.6 * 4096))],
)
def test_read_write_slices(tmp_path, xp, gds_threshold, nthreads, tasksize, start, end):
    """Read and write different slices"""

    with kvikio.defaults.set({"num_threads": nthreads, "task_size": tasksize}):
        filename = tmp_path / "test-file"
        a = xp.arange(10 * 4096)  # 10 page-sizes
        b = a.copy()
        a[start:end] = 42
        with kvikio.CuFile(filename, "w") as f:
            assert f.write(a[start:end]) == a[start:end].nbytes
        with kvikio.CuFile(filename, "r") as f:
            assert f.read(b[start:end]) == b[start:end].nbytes
        xp.testing.assert_array_equal(a, b)


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


def test_no_current_cuda_context(tmp_path, xp, gds_threshold):
    """Test IO when CUDA context is current"""
    filename = tmp_path / "test-file"
    a = xp.arange(100)
    b = xp.empty_like(a)

    with kvikio.CuFile(filename, "w+") as f:
        with with_no_cuda_context():
            f.write(a)
        f.read(b)
    xp.testing.assert_array_equal(a, b)


@pytest.mark.skipif(
    cupy.cuda.runtime.getDeviceCount() < 2, reason="requires multiple GPUs"
)
def test_multiple_gpus(tmp_path, xp, gds_threshold):
    """Test IO from two different GPUs"""
    filename = tmp_path / "test-file"

    with kvikio.defaults.set({"num_threads": 10, "task_size": 10}):
        # Allocate an array on each device
        with cupy.cuda.Device(0):
            a0 = xp.arange(200)
        with cupy.cuda.Device(1):
            a1 = xp.zeros(200, dtype=a0.dtype)

        # Test when the device match the allocation
        with kvikio.CuFile(filename, "w") as f:
            with cupy.cuda.Device(0):
                assert f.write(a0) == a0.nbytes
        with kvikio.CuFile(filename, "r") as f:
            with cupy.cuda.Device(1):
                assert f.read(a1) == a1.nbytes
        assert bytes(a0) == bytes(a1)

        # Test when the device doesn't match the allocation
        with kvikio.CuFile(filename, "w") as f:
            with cupy.cuda.Device(1):
                assert f.write(a0) == a0.nbytes
        with kvikio.CuFile(filename, "r") as f:
            with cupy.cuda.Device(0):
                assert f.read(a1) == a1.nbytes
        assert bytes(a0) == bytes(a1)


@pytest.mark.parametrize("size", [1, 10, 100, 1000])
@pytest.mark.parametrize("tasksize", [1, 10, 100, 1000])
@pytest.mark.parametrize("buffer_size", [1, 10, 100, 1000])
def test_different_bounce_buffer_sizes(tmp_path, size, tasksize, buffer_size):
    """Test different bounce buffer sizes"""
    filename = tmp_path / "test-file"
    with kvikio.defaults.set(
        {
            "compat_mode": kvikio.CompatMode.ON,
            "num_threads": 10,
            "bounce_buffer_size": buffer_size,
        }
    ):
        with kvikio.CuFile(filename, "w+") as f:
            a = cupy.arange(size)
            b = cupy.empty_like(a)
            f.write(a)
            assert f.read(b) == b.nbytes
            cupy.testing.assert_array_equal(a, b)


def test_bounce_buffer_free(tmp_path):
    """Test freeing the bounce buffer allocations"""
    filename = tmp_path / "test-file"
    kvikio.buffer.bounce_buffer_free()
    with kvikio.defaults.set({"compat_mode": kvikio.CompatMode.ON, "num_threads": 1}):
        with kvikio.CuFile(filename, "w") as f:
            with kvikio.defaults.set({"bounce_buffer_size": 1024}):
                # Notice, since the bounce buffer size is only checked when the buffer
                # is used, we populate the bounce buffer in between we clear it.
                f.write(cupy.arange(10))
                assert kvikio.buffer.bounce_buffer_free() == 1024
                assert kvikio.buffer.bounce_buffer_free() == 0
                f.write(cupy.arange(10))
            with kvikio.defaults.set({"bounce_buffer_size": 2048}):
                f.write(cupy.arange(10))
                assert kvikio.buffer.bounce_buffer_free() == 2048
                assert kvikio.buffer.bounce_buffer_free() == 0


def test_get_page_cache_info(tmp_path):
    """Test getting the page cache information for a file"""
    with pytest.raises(RuntimeError, match="Unable to open file"):
        nonexistent_file = tmp_path / "nonexistent_file"
        kvikio.get_page_cache_info(nonexistent_file)

    with pytest.raises(ValueError):
        invalid_argument = 123.456
        kvikio.get_page_cache_info(invalid_argument)

    with pytest.raises(ValueError):
        invalid_argument = ["path_in_a_list"]
        kvikio.get_page_cache_info(invalid_argument)

    test_file = tmp_path / "test_file"
    with kvikio.CuFile(test_file, "w") as f:
        num_elements = 10000
        f.write(cupy.linspace(1.0, float(num_elements), num=num_elements))

    # Pass an os.PathLike argument
    arg = test_file
    assert isinstance(arg, os.PathLike)
    num_pages_in_page_cache, num_pages = kvikio.get_page_cache_info(arg)
    assert num_pages_in_page_cache >= 0 and num_pages_in_page_cache <= num_pages

    # Pass a string argument
    arg = str(test_file)
    assert isinstance(arg, str)
    num_pages_in_page_cache, num_pages = kvikio.get_page_cache_info(arg)
    assert num_pages_in_page_cache >= 0 and num_pages_in_page_cache <= num_pages

    # Pass a file descriptor argument
    with open(test_file, "rb") as f:
        arg = f.fileno()
        assert isinstance(arg, int)
        num_pages_in_page_cache, num_pages = kvikio.get_page_cache_info(arg)
        assert num_pages_in_page_cache >= 0 and num_pages_in_page_cache <= num_pages

    # Pass a file object argument
    with open(test_file, "rb") as f:
        arg = f
        assert isinstance(arg, io.IOBase)
        num_pages_in_page_cache, num_pages = kvikio.get_page_cache_info(arg)
        assert num_pages_in_page_cache >= 0 and num_pages_in_page_cache <= num_pages
