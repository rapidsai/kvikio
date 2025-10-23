# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os

import cupy
import pytest

import kvikio
import kvikio.defaults


def check_bit_flags(x: int, y: int) -> bool:
    """Check that the bits set in `y` is also set in `x`"""
    return x & y == y


@pytest.mark.parametrize("size", [1, 10, 100, 1000, 1024, 4096, 4096 * 10])
def test_read_write(tmp_path, size):
    """Test basic read/write"""
    filename = tmp_path / "test-file"

    stream = cupy.cuda.Stream()

    # Write file
    a = cupy.arange(size)
    f = kvikio.CuFile(filename, "w")
    assert not f.closed
    assert check_bit_flags(f.open_flags(), os.O_WRONLY)
    assert f.raw_write_async(a, stream.ptr).check_bytes_done() == a.nbytes

    # Try to read file opened in write-only mode
    # POSIX read would yield the error "Operation not permitted"
    # cuFile read would yield the error "unsupported file open flags"
    with pytest.raises(
        RuntimeError, match="Operation not permitted|unsupported file open flags"
    ):
        # The exception is raised when we call the raw_read_async API.
        future_stream = f.raw_read_async(a, stream.ptr)
        future_stream.check_bytes_done()

    # Close file
    f.close()
    assert f.closed

    # Read file into a new array and compare
    b = cupy.empty_like(a)
    c = cupy.empty_like(a)
    f = kvikio.CuFile(filename, "r+")
    assert check_bit_flags(f.open_flags(), os.O_RDWR)

    future_stream = f.raw_read_async(b, stream.ptr)
    future_stream2 = f.raw_write_async(b, stream.ptr)
    future_stream3 = f.raw_read_async(c, stream.ptr)
    assert (
        future_stream.check_bytes_done()
        == future_stream2.check_bytes_done()
        == future_stream3.check_bytes_done()
        == b.nbytes
    )
    cupy.testing.assert_array_equal(a, b)
    cupy.testing.assert_array_equal(a, c)
