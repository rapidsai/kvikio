# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math
import socket
from typing import Any

import cupy
import numpy.typing as npt


def localhost() -> str:
    return "127.0.0.1"


def find_free_port(host: str = localhost()) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, 0))
        _, port = s.getsockname()
    return port


def empty_page_aligned(
    shape: int | tuple[int, ...],
    dtype: npt.DTypeLike = cupy.float64,
    page_size: int = 4096,
) -> cupy.ndarray:
    """Allocate an uninitialized page-aligned CuPy array.

    Parameters
    ----------
    shape: Shape of the array. Can be an integer for 1D arrays or a tuple of integers
        for multi-dimensional arrays.
    dtype: Data type of the array elements. Defaults to ``cupy.float64``.
    page_size: Page size in bytes. Defaults to 4096 (4KB).

    Returns
    -------
        An uninitialized CuPy array with page-aligned underlying memory.
    """
    resolved_dtype: cupy.dtype[Any] = cupy.dtype(dtype)
    if isinstance(shape, int):
        shape = (shape,)

    size = math.prod(shape) * resolved_dtype.itemsize

    # Over-allocate for alignment
    backing_mem = cupy.cuda.Memory(size + page_size - 1)
    aligned_ptr = (backing_mem.ptr + page_size - 1) & ~(page_size - 1)
    aligned_memptr = cupy.cuda.MemoryPointer(backing_mem, aligned_ptr - backing_mem.ptr)

    return cupy.ndarray(shape, dtype=dtype, memptr=aligned_memptr)


def arange_page_aligned(
    stop: int, dtype: npt.DTypeLike = cupy.float64, page_size: int = 4096
) -> cupy.ndarray:
    """Create a page-aligned CuPy array with incremental values from 0 to ``stop - 1``.

    Parameters
    ----------
    stop: Number of elements. The array will contain values from 0 to ``stop - 1``
        (exclusive upper bound).
    dtype: Data type of the array elements. Defaults to ``cupy.float64``.
    page_size: Page size in bytes. Defaults to 4096 (4KB).

    Returns
    -------
    A CuPy array with values ``[0, 1, ..., stop-1]`` and page-aligned underlying memory.
    """
    arr = empty_page_aligned(stop, dtype=dtype, page_size=page_size)
    arr[:] = cupy.arange(stop, dtype=dtype)
    return arr
