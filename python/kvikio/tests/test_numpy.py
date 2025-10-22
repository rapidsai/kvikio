# SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import pytest

from kvikio.numpy import LikeWrapper, tofile


@pytest.mark.parametrize("dtype", ["u1", "int64", "float32", "float64"])
def test_tofile(tmp_path, xp, dtype):
    """Test tofile()"""

    filepath = str(tmp_path / "test_tofile")
    src = xp.arange(100, dtype=dtype)
    tofile(src, filepath)

    dst = xp.fromfile(filepath, dtype=dtype)
    xp.testing.assert_array_equal(src, dst)

    tofile(src[::2], filepath)
    dst = xp.fromfile(filepath, dtype=dtype)
    xp.testing.assert_array_equal(src[::2], dst)


@pytest.mark.parametrize("dtype", ["u1", "int64", "float32", "float64"])
def test_fromfile(tmp_path, xp, dtype):
    """Test NumPy's and CuPy's fromfile() with LikeWrapper"""

    filepath = str(tmp_path / "test_fromfile")
    src = xp.arange(100, dtype=dtype)
    src.tofile(filepath)
    like = LikeWrapper(like=xp.empty(()))
    dst = xp.fromfile(filepath, dtype, like=like)
    xp.testing.assert_array_equal(src, dst)
    dst = xp.fromfile(filepath, dtype=dtype, like=like)
    xp.testing.assert_array_equal(src, dst)
    dst = xp.fromfile(file=filepath, dtype=dtype, like=like)
    xp.testing.assert_array_equal(src, dst)
    dst = xp.fromfile(file=filepath, dtype=dtype, count=100 - 42, like=like)
    xp.testing.assert_array_equal(src[:-42], dst)
    dst = xp.fromfile(file=filepath, dtype=dtype, offset=src.itemsize, like=like)
    xp.testing.assert_array_equal(src[1:], dst)
    dst = xp.fromfile(file=filepath, dtype=dtype, offset=1, count=10, like=like)
    assert len(dst) == 10

    # Test non-divisible offset
    dst = xp.fromfile(file=filepath, dtype="u1", offset=7, like=like)
    xp.testing.assert_array_equal(src.view(dtype="u1")[7:], dst)

    filepath = str(tmp_path / "test_fromfile")
    with open(filepath, mode="rb") as f:
        dst = xp.fromfile(file=f, dtype=dtype, like=like)
        xp.testing.assert_array_equal(src, dst)


def test_fromfile_error(tmp_path, xp):
    filepath = str(tmp_path / "test_fromfile")
    src = xp.arange(1, dtype="u1")
    src.tofile(filepath)
    like = LikeWrapper(like=src)

    with pytest.raises(FileNotFoundError, match="no file"):
        xp.fromfile("no file", like=like)

    with pytest.raises(NotImplementedError, match="Non-default value of the `sep`"):
        xp.fromfile(file=filepath, sep=",", like=like)

    with pytest.raises(ValueError, match="[Nn]egative dimensions are not allowed"):
        xp.fromfile(file=filepath, like=like, count=-42)
