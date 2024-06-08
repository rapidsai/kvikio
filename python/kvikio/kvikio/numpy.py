# Copyright (c) 2022-2023, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.

import io
import os
import os.path
from typing import Protocol, Union, runtime_checkable

import numpy as np
from numpy.typing import ArrayLike, DTypeLike

import kvikio


@runtime_checkable
class FileLike(Protocol):
    """File like object that represent a OS-level file"""

    def fileno(self) -> int:
        ...

    @property
    def name(self) -> str:
        ...


class LikeWrapper:
    """Wrapper for NumPy's `like` argument introduced in NumPy v1.20

    Wraps an array-like instance in order to seamlessly utilize KvikIO.

    Examples
    --------
    Read file into a NumPy array:

    >>> np.arange(10).tofile("/tmp/myfile")
    >>> np.fromfile("/tmp/myfile", dtype=int)
    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    >>> type(_)
    <class 'numpy.ndarray'>

    Read file into a CuPy array using the like argument. The file is read
    directly into device memory using GDS if available:

    >>> import cupy
    >>> np.fromfile("/tmp/myfile", dtype=int, like=cupy.empty(()))
    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    >>> type(_)
    <class 'cupy._core.core.ndarray'>

    We can also use CuPy's fromfile function:

    >>> cupy.fromfile("/tmp/myfile", dtype=int, like=cupy.empty(()))
    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    >>> type(_)
    <class 'cupy._core.core.ndarray'>
    """

    def __init__(self, like: ArrayLike) -> None:
        self._like = like

    def __array_function__(self, func, types, args, kwargs):
        if func is not np.fromfile:
            return func(*args, **kwargs)
        return fromfile(*args, like=self._like, **kwargs)


def fromfile(
    file: Union[str, os.PathLike, io.FileIO],
    dtype: DTypeLike = float,
    count: int = -1,
    sep: str = "",
    offset: int = 0,
    *,
    like: ArrayLike = None,
) -> ArrayLike:
    """Construct an array from file using KvikIO

    Overload `numpy.fromfile` to use KvikIO.

    Parameters
    ----------
    file : FileLike or str or PathLike
        Open file object or filename.
    dtype : data-type
        Data type of the returned array.
        For binary files, it is used to determine the size and byte-order
        of the items in the file.
        Most builtin numeric types are supported and extension types may be supported.
    count : int
        Number of items to read. `-1` means all items (i.e., the complete file).
    sep : str
        Empty ("") separator means the file should be treated as binary. Any other
        value is not supported and will raise NotImplementedError.
    offset : int
        The offset (in bytes) from the file's current position. Defaults to 0.
        Only permitted for binary files.
    like : array_like, optional
        Reference object to allow the creation of arrays which are not
        NumPy arrays.

    Examples
    --------
    Read file into a NumPy array:

    >>> np.arange(10).tofile("/tmp/myfile")
    >>> fromfile("/tmp/myfile", dtype=int)
    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    >>> type(_)
    <class 'numpy.ndarray'>

    Read file into a CuPy array using the like argument. The file is read
    directly into device memory using GDS if available:

    >>> import cupy
    >>> fromfile("/tmp/myfile", dtype=int, like=cupy.empty(()))
    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    >>> type(_)
    <class 'cupy._core.core.ndarray'>
    """

    if sep != "":
        raise NotImplementedError(
            "Non-default value of the `sep` argument is not supported"
        )

    if isinstance(file, FileLike):
        filepath = file.name
    else:
        filepath = str(file)

    nbytes = os.path.getsize(filepath)
    itemsize = np.dtype(dtype).itemsize
    count = nbytes if count == -1 else count
    # Notice, Numpy truncate to filesize silently.
    count = min(count, (nbytes - offset) // itemsize)

    ret = np.empty_like(like, shape=(count,), dtype=dtype)
    with kvikio.CuFile(filepath, "r") as f:
        f.read(ret, file_offset=offset)
    return ret


def tofile(
    ary: ArrayLike,
    file: Union[str, os.PathLike, io.FileIO],
) -> None:
    """Write array to a file using KvikIO.

    Overload `numpy.fromfile` to use KvikIO.

    Data is always written in 'C' order, independent of the order of `a`.
    The data produced by this method can be recovered using the function
    fromfile().

    Parameters
    ----------
    ary : array_like
        Data to write.
    file : FileLike or str or PathLike
        Open file object or filename.

    Notes
    -----
    This is a convenience function for quick storage of array data.
    Information on endianness and precision is lost, so this method is not a
    good choice for files intended to archive data or transport data between
    machines with different endianness.

    When file is a file object, array contents are directly written to the
    file, bypassing the file object's `write` method. As a result, tofile
    cannot be used with files objects supporting compression (e.g., GzipFile)
    or file-like objects that do not support `fileno()` (e.g., BytesIO).
    """
    if isinstance(file, FileLike):
        filepath = file.name
    else:
        filepath = str(file)
    with kvikio.CuFile(filepath, "w") as f:
        f.write(np.ascontiguousarray(ary, like=ary))  # type: ignore
