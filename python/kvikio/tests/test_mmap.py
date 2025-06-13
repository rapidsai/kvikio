# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.

import os

import pytest

import kvikio.defaults

cupy = pytest.importorskip("cupy")
numpy = pytest.importorskip("numpy")


def test_no_file(tmp_path):
    with pytest.raises(RuntimeError, match=r".*Unable to open file.*"):
        nonexistent_file = tmp_path / "nonexistent_file"
        kvikio.MmapHandle(nonexistent_file)


def test_invalid_file_open_flag(tmp_path):
    filename = tmp_path / "read-only-test-file"
    expected_data = numpy.arange(1024)
    expected_data.tofile(filename)

    with pytest.raises(ValueError, match=r".*Unknown file open flag.*"):
        kvikio.MmapHandle(filename, "")

    with pytest.raises(ValueError, match=r".*Unknown file open flag.*"):
        kvikio.MmapHandle(filename, "z")


def test_constructor_invalid_range(tmp_path, xp):
    filename = tmp_path / "read-only-test-file"
    test_data = xp.arange(1024 * 1024)
    test_data.tofile(filename)

    with pytest.raises(IndexError, match=r".*Offset is past the end of file.*"):
        kvikio.MmapHandle(filename, "r", None, test_data.nbytes * 2)

    with pytest.raises(IndexError, match=r".*Mapped region is past the end of file.*"):
        kvikio.MmapHandle(filename, "r", test_data.nbytes * 2)

    with pytest.raises(ValueError, match=r".*Mapped region should not be zero byte.*"):
        kvikio.MmapHandle(filename, "r", 0)


def test_read_invalid_range(tmp_path, xp):
    filename = tmp_path / "read-only-test-file"
    test_data = xp.arange(1024 * 1024)
    test_data.tofile(filename)
    output_data = xp.zeros_like(test_data)

    initial_size = 1024
    initial_file_offset = 512

    with pytest.raises(IndexError, match=r".*Offset is past the end of file.*"):
        mmap_handle = kvikio.MmapHandle(
            filename, "r", initial_size, initial_file_offset
        )
        mmap_handle.read(output_data, initial_size, test_data.nbytes)

    with pytest.raises(IndexError, match=r".*Read is out of bound.*"):
        mmap_handle = kvikio.MmapHandle(
            filename, "r", initial_size, initial_file_offset
        )
        mmap_handle.read(output_data, initial_size, initial_file_offset - 128)

    with pytest.raises(ValueError, match=r".*Read size must be greater than 0.*"):
        mmap_handle = kvikio.MmapHandle(
            filename, "r", initial_size, initial_file_offset
        )
        mmap_handle.read(output_data, 0, initial_file_offset)

    with pytest.raises(IndexError, match=r".*Read is out of bound.*"):
        mmap_handle = kvikio.MmapHandle(
            filename, "r", initial_size, initial_file_offset
        )
        mmap_handle.read(output_data, initial_size + 128, initial_file_offset)


@pytest.mark.parametrize("num_elements_to_read", [None, 10, 9999])
@pytest.mark.parametrize("num_elements_to_skip", [0, 10, 100, 1000, 9999])
def test_read_seq(tmp_path, xp, num_elements_to_read, num_elements_to_skip):
    filename = tmp_path / "read-only-test-file"
    test_data = xp.arange(1024 * 1024)
    test_data.tofile(filename)

    if num_elements_to_read is None:
        initial_size = None
        actual_num_elements_to_read = int(
            os.path.getsize(filename) / test_data.itemsize
        )
    else:
        initial_size = num_elements_to_read * test_data.itemsize
        actual_num_elements_to_read = num_elements_to_read

    initial_file_offset = num_elements_to_skip * test_data.itemsize
    expected_data = test_data[
        num_elements_to_skip : (num_elements_to_skip + actual_num_elements_to_read)
    ]
    actual_data = xp.zeros_like(expected_data)

    mmap_handle = kvikio.MmapHandle(filename, "r", initial_size, initial_file_offset)
    read_size = mmap_handle.read(actual_data, initial_size, initial_file_offset)

    assert read_size == expected_data.nbytes
    xp.testing.assert_array_equal(actual_data, expected_data)


@pytest.mark.parametrize("num_elements_to_read", [None, 10, 9999])
@pytest.mark.parametrize("num_elements_to_skip", [0, 10, 100, 1000, 9999])
@pytest.mark.parametrize("mmap_task_size", [0, 1024, 12345])
def test_read_parallel(
    tmp_path, xp, num_elements_to_read, num_elements_to_skip, mmap_task_size
):
    filename = tmp_path / "read-only-test-file"
    test_data = xp.arange(1024 * 1024)
    test_data.tofile(filename)

    if num_elements_to_read is None:
        initial_size = None
        actual_num_elements_to_read = int(
            os.path.getsize(filename) / test_data.itemsize
        )
    else:
        initial_size = num_elements_to_read * test_data.itemsize
        actual_num_elements_to_read = num_elements_to_read

    initial_file_offset = num_elements_to_skip * test_data.itemsize
    expected_data = test_data[
        num_elements_to_skip : (num_elements_to_skip + actual_num_elements_to_read)
    ]
    actual_data = xp.zeros_like(expected_data)

    with kvikio.defaults.set("mmap_task_size", mmap_task_size):
        mmap_handle = kvikio.MmapHandle(
            filename, "r", initial_size, initial_file_offset
        )
        fut = mmap_handle.pread(
            actual_data, initial_size, initial_file_offset, mmap_task_size
        )

        assert fut.get() == expected_data.nbytes
        xp.testing.assert_array_equal(actual_data, expected_data)


def test_read_with_default_arguments(tmp_path, xp):
    filename = tmp_path / "read-only-test-file"
    expected_data = xp.arange(1024 * 1024)
    expected_data.tofile(filename)
    actual_data = xp.zeros_like(expected_data)

    mmap_handle = kvikio.MmapHandle(filename, "r")

    read_size = mmap_handle.read(actual_data)
    assert read_size == expected_data.nbytes
    xp.testing.assert_array_equal(actual_data, expected_data)

    fut = mmap_handle.pread(actual_data)
    assert fut.get() == expected_data.nbytes
    xp.testing.assert_array_equal(actual_data, expected_data)


def test_closed_handle(tmp_path, xp):
    filename = tmp_path / "read-only-test-file"
    expected_data = xp.arange(1024 * 1024)
    expected_data.tofile(filename)
    actual_data = xp.zeros_like(expected_data)

    mmap_handle = kvikio.MmapHandle(filename, "r")
    mmap_handle.close()

    assert mmap_handle.closed()
    assert mmap_handle.file_size() == 0

    with pytest.raises(RuntimeError, match=r".*Cannot read from a closed MmapHandle.*"):
        mmap_handle.read(actual_data)

    with pytest.raises(RuntimeError, match=r".*Cannot read from a closed MmapHandle.*"):
        mmap_handle.pread(actual_data)
