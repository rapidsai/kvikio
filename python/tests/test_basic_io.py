import os
import random

import pytest

import cufile

cupy = pytest.importorskip("cupy")


def test_read_write(tmp_path):
    """Test basic read/write"""
    filename = tmp_path / "test-file"
    print(filename)

    # Write file
    a = cupy.arange(100)
    f = cufile.CuFile(filename, "w")
    assert not f.closed
    assert f.open_flags() & (os.O_WRONLY | os.O_DIRECT | os.O_CLOEXEC)
    assert f.write(a) == a.nbytes

    # Try to read file opened in write-only mode
    with pytest.raises(RuntimeError, match="unsupported file open flags"):
        f.read(a)

    # Close file
    f.close()
    assert f.closed

    # Read file into a new array and compare
    b = cupy.empty_like(a)
    f = cufile.CuFile(filename, "r")
    assert f.open_flags() & (os.O_RDONLY | os.O_DIRECT | os.O_CLOEXEC)
    f.read(b)
    assert all(a == b)


def test_write_in_offsets(tmp_path):
    """Write to files in chunks"""
    filename = tmp_path / "test-file"

    a = cupy.arange(200)
    f = cufile.CuFile(filename, "w")

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
    b = cupy.empty_like(a)
    f = cufile.CuFile(filename, "r")
    f.read(b)
    assert all(a == b)
