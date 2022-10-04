# Copyright (c) 2021-2022, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.


import pytest

from kvikio.numpy import FromFile


@pytest.mark.parametrize("xp", ["numpy", "cupy"])
@pytest.mark.parametrize("dtype", ["u1", "int64", "float32", "float64"])
def test_from_file(tmp_path, xp, dtype):
    """Test NumPy's and CuPy's fromfile() with FromFile"""

    xp = pytest.importorskip(xp)
    filepath = str(tmp_path / "test_from_file")
    src = xp.arange(100, dtype=dtype)
    src.tofile(filepath)
    like = FromFile(meta_array=xp.empty(()))
    dst = xp.fromfile(filepath, dtype, like=like)
    xp.testing.assert_array_equal(src, dst)
    dst = xp.fromfile(filepath, dtype=dtype, like=like)
    xp.testing.assert_array_equal(src, dst)
    dst = xp.fromfile(file=filepath, dtype=dtype, like=like)
    xp.testing.assert_array_equal(src, dst)

    filepath = str(tmp_path / "test_from_file")
    with open(filepath, mode="rb") as f:
        dst = xp.fromfile(file=f, dtype=dtype, like=like)
        xp.testing.assert_array_equal(src, dst)


@pytest.mark.parametrize("xp", ["numpy", "cupy"])
def test_from_file_error(tmp_path, xp):
    xp = pytest.importorskip(xp)
    filepath = str(tmp_path / "test_from_file")
    src = xp.arange(1, dtype="u1")
    src.tofile(filepath)

    with pytest.raises(FileNotFoundError, match="no file"):
        xp.fromfile("no file", like=FromFile(meta_array=src))

    with pytest.raises(ValueError, match="not divisible with dtype"):
        xp.fromfile(file=filepath, like=FromFile(meta_array=src))
