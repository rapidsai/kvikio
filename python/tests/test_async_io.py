import os
import cupy
import pytest
import torch

import kvikio
import kvikio.defaults

import torch


def check_bit_flags(x: int, y: int) -> bool:
    """Check that the bits set in `y` is also set in `x`"""
    return x & y == y


@pytest.mark.parametrize("size", [1, 10, 100, 1000, 1024, 4096, 4096 * 10])
@pytest.mark.parametrize("nthreads", [1, 3, 4, 16])
@pytest.mark.parametrize("tasksize", [199, 1024])
def test_read_write(tmp_path, gds_threshold, size, nthreads, tasksize):
    """Test basic read/write"""
    filename = tmp_path / "test-file"

    stream = torch.cuda.stream(torch.cuda.Stream())
    stream_ptr = stream.stream.cuda_stream

    with kvikio.defaults.set_num_threads(nthreads):
        with kvikio.defaults.set_task_size(tasksize):
            # Write file
            a = cupy.arange(size)
            f = kvikio.CuFile(filename, "w")
            assert not f.closed
            assert check_bit_flags(f.open_flags(), os.O_WRONLY)
            assert f.write_async(a, stream_ptr) == a.nbytes

            # Try to read file opened in write-only mode
            with pytest.raises(
                RuntimeError, match="unsupported file open flags"
            ):
                f.read_async(a, stream_ptr)

            # Close file
            f.close()
            assert f.closed

            # Read file into a new array and compare
            b = cupy.empty_like(a)
            f = kvikio.CuFile(filename, "r")
            assert check_bit_flags(f.open_flags(), os.O_RDONLY)
            assert f.read_async(b, stream_ptr) == b.nbytes
            assert all(a == b)
