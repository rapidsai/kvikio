# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.


import math

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from legate.core import get_legate_runtime
from legate_kvikio.zarr import read_array, write_array

num = pytest.importorskip("cunumeric")
zarr = pytest.importorskip("zarr")


shape_chunks = (
    "shape,chunks",
    [
        ((2,), (2,)),
        ((5,), (2,)),
        ((4, 2), (2, 2)),
        ((2, 4), (2, 2)),
        ((2, 3), (3, 2)),
        ((4, 3, 2, 1), (1, 2, 3, 4)),
    ],
)


@pytest.mark.parametrize(*shape_chunks)
@pytest.mark.parametrize("dtype", ["u1", "u8", "f8"])
def test_write_array(tmp_path, shape, chunks, dtype):
    """Test write of a Zarr array"""
    a = num.arange(math.prod(shape), dtype=dtype).reshape(shape)

    write_array(ary=a, dirpath=tmp_path, chunks=chunks)
    get_legate_runtime().issue_execution_fence(block=True)

    b = zarr.open_array(tmp_path, mode="r")
    assert_array_equal(a, b)


@pytest.mark.parametrize(*shape_chunks)
@pytest.mark.parametrize("dtype", ["u1", "u8", "f8"])
def test_read_array(tmp_path, shape, chunks, dtype):
    """Test read of a Zarr array"""
    a = np.arange(math.prod(shape), dtype=dtype).reshape(shape)
    zarr.open_array(tmp_path, mode="w", shape=shape, chunks=chunks, compressor=None)[
        ...
    ] = a

    b = read_array(dirpath=tmp_path)
    assert_array_equal(a, b)
