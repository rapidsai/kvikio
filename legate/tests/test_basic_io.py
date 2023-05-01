# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.


import math

import pytest

from legate.core import get_legate_runtime
from legate_kvikio import CuFile
from legate_kvikio.tile import read_tiles, write_tiles

num = pytest.importorskip("cunumeric")


def fence(*, block: bool):
    """Shorthand for a Legate fence"""
    get_legate_runtime().issue_execution_fence(block=block)


@pytest.mark.parametrize("size", [1, 10, 100, 1000, 1024, 4096, 4096 * 10])
def test_read_write(tmp_path, size):
    """Test basic read/write"""
    filename = tmp_path / "test-file"
    a = num.arange(size)
    f = CuFile(filename, "w")
    f.write(a)
    assert not f.closed
    fence(block=True)

    # Try to read file opened in write-only mode
    with pytest.raises(ValueError, match="Cannot read a file opened with flags"):
        f.read(a)

    # Close file
    f.close()
    assert f.closed

    # Read file into a new array and compare
    b = num.empty_like(a)
    f = CuFile(filename, "r")
    f.read(b)
    assert all(a == b)


def test_file_handle_context(tmp_path):
    """Open a CuFile in a context"""
    filename = tmp_path / "test-file"
    a = num.arange(200)
    b = num.empty_like(a)
    with CuFile(filename, "w+") as f:
        assert not f.closed
        f.write(a)
        fence(block=False)
        f.read(b)
        assert all(a == b)
    assert f.closed


@pytest.mark.parametrize(
    "start,end",
    [
        (0, 10),
        (1, 10),
        (0, 10 * 4096),
        (1, int(1.3 * 4096)),
        (int(2.1 * 4096), int(5.6 * 4096)),
    ],
)
def test_read_write_slices(tmp_path, start, end):
    """Read and write different slices"""

    filename = tmp_path / "test-file"
    a = num.arange(10 * 4096)  # 10 page-sizes
    b = a.copy()
    a[start:end] = 42
    with CuFile(filename, "w") as f:
        f.write(a[start:end])
    fence(block=True)
    with CuFile(filename, "r") as f:
        f.read(b[start:end])
    assert all(a == b)


@pytest.mark.parametrize(
    "shape,tile_shape", [((2,), (3,)), ((2, 2), (3, 2)), ((2, 3), (2, 2))]
)
def test_read_write_tiles_error(tmp_path, shape, tile_shape):
    with pytest.raises(ValueError, match="must be divisible"):
        write_tiles(ary=num.ones(shape), dirpath=tmp_path, tile_shape=tile_shape)
    with pytest.raises(ValueError, match="must be divisible"):
        read_tiles(ary=num.ones(shape), dirpath=tmp_path, tile_shape=tile_shape)


@pytest.mark.parametrize(
    "shape,tile_shape,tile_start",
    [
        ((2,), (2,), (1,)),
        ((4,), (2,), (0,)),
        ((4, 2), (2, 2), (1, 2)),
        ((2, 4), (2, 2), (2, 1)),
    ],
)
def test_read_write_tiles(tmp_path, shape, tile_shape, tile_start):
    a = num.arange(math.prod(shape)).reshape(shape)
    write_tiles(ary=a, dirpath=tmp_path, tile_shape=tile_shape, tile_start=tile_start)
    fence(block=True)
    b = num.empty_like(a)
    read_tiles(ary=b, dirpath=tmp_path, tile_shape=tile_shape, tile_start=tile_start)
    assert (a == b).all()
