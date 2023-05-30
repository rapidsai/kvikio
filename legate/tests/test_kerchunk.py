# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.


import math

import numpy as np
import pytest
from numpy.testing import assert_array_equal

hdf5_read = pytest.importorskip("legate_kvikio.kerchunk").hdf5_read

num = pytest.importorskip("cunumeric")

shape_chunks = (
    "shape,chunks",
    [
        ((2,), (2,)),
        ((5,), (2,)),
        ((4, 2), (2, 2)),
        ((2, 4), (2, 2)),
        ((2, 3), (2, 2)),
        ((5, 4, 3, 2), (2, 2, 2, 2)),
    ],
)


@pytest.mark.parametrize(*shape_chunks)
@pytest.mark.parametrize("dtype", ["u1", "u8", "f8"])
def test_hdf5_read_array(tmp_path, shape, chunks, dtype):
    h5py = pytest.importorskip("h5py")

    filename = tmp_path / "test-file.hdf5"
    a = np.arange(math.prod(shape), dtype=dtype).reshape(shape)
    with h5py.File(filename, "w") as f:
        f.create_dataset("mydataset", chunks=chunks, data=a)

    b = hdf5_read(filename, dataset_name="mydataset")
    assert_array_equal(a, b)
