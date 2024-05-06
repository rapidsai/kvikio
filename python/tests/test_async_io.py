# Copyright (c) 2022-2024, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.

import os

import cupy
import pytest

import kvikio
import kvikio.defaults


def check_bit_flags(x: int, y: int) -> bool:
    """Check that the bits set in `y` is also set in `x`"""
    return x & y == y


@pytest.mark.parametrize("size", [1, 10, 100, 1000, 1024, 4096, 4096 * 10])
@pytest.mark.parametrize("nthreads", [1, 3, 4, 16])
@pytest.mark.parametrize("tasksize", [199, 1024])
def test_read_write(tmp_path, gds_threshold, size, nthreads, tasksize):
    """Test basic read/write"""
    filename = tmp_path / "test-file"

    stream = cupy.cuda.Stream()

    with kvikio.defaults.set_num_threads(nthreads):
        with kvikio.defaults.set_task_size(tasksize):
            # Write file
            a = cupy.arange(size)
            f = kvikio.CuFile(filename, "w")
            assert not f.closed
            assert check_bit_flags(f.open_flags(), os.O_WRONLY)
            assert f.raw_write_async(a, stream.ptr).check_bytes_done() == a.nbytes

            # Try to read file opened in write-only mode
            with pytest.raises(RuntimeError, match="unsupported file open flags"):
                f.raw_read_async(a, stream.ptr).check_bytes_done()

            # Close file
            f.close()
            assert f.closed

            # Read file into a new array and compare
            b = cupy.empty_like(a)
            f = kvikio.CuFile(filename, "r")
            assert check_bit_flags(f.open_flags(), os.O_RDONLY)
            assert f.raw_read_async(b, stream.ptr).check_bytes_done() == b.nbytes
            assert all(a == b)
