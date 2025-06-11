# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.

import os

import pytest

import kvikio.defaults

cupy = pytest.importorskip("cupy")
numpy = pytest.importorskip("numpy")


@pytest.mark.parametrize("size", [1, 10, 100, 1000, 1024, 4096, 4096 * 10])
@pytest.mark.parametrize("num_threads", [1, 3, 4, 16])
@pytest.mark.parametrize("mmap_task_size", [1024])
def test_read(tmp_path, xp, size, num_threads, mmap_task_size):
    """Test mmap read"""
    filename = tmp_path / "test-file"

    with kvikio.defaults.set(
        {"num_threads": num_threads, "mmap_task_size": mmap_task_size}
    ):
        a = numpy.arange(size)
        a.tofile(filename)
        os.sync()

        b = xp.empty_like(a)
        mmap_handle = kvikio.MmapHandle(filename, "r")
        assert mmap_handle.read(b) == b.nbytes
        xp.testing.assert_array_equal(a, b)
