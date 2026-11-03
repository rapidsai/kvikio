# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math
import os
import socket
import sys
from collections.abc import Callable, Sequence
from typing import Any

import cupy
import numpy.typing as npt
import pytest


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
    shape: int or tuple of ints
        Shape of the array. Can be an integer for 1D arrays or a tuple of integers
        for multi-dimensional arrays.
    dtype: npt.DTypeLike
        Data type of the array elements. Defaults to ``cupy.float64``.
    page_size: int
        Page size in bytes. Defaults to 4096 (4KB).

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
    stop: int
        Number of elements. The array will contain values from 0 to ``stop - 1``
        (exclusive upper bound).
    dtype: npt.DTypeLike
        Data type of the array elements. Defaults to ``cupy.float64``.
    page_size: int
        Page size in bytes. Defaults to 4096 (4KB).

    Returns
    -------
    A CuPy array with values ``[0, 1, ..., stop-1]`` and page-aligned underlying memory.
    """
    arr_aligned = empty_page_aligned(stop, dtype=dtype, page_size=page_size)
    arr_unaligned = cupy.arange(stop, dtype=dtype)
    cupy.copyto(arr_aligned, arr_unaligned)
    return arr_aligned


def rerun_in_new_process(
    test_file: str, env: dict[str, str], pytest_args: Sequence[str] = ()
) -> Callable[..., None]:
    """Make a test that reruns a test file in a new process with extra env vars.

    Assign the result to a module-level name starting with ``test_`` so that pytest
    collects it, for example
    ``test_foo_under_bar = utils.rerun_in_new_process(__file__, {"BAR": "1"})``.

    The rerun is skipped in a process that already has any of the ``env`` variables
    set. This serves to keep the new process from starting reruns of its own.

    Parameters
    ----------
    test_file
        Path of the test file to rerun, usually ``__file__``.
    env
        Environment variables to add for the new process.
    pytest_args
        Extra arguments for the new pytest run, such as ``["-k", "name"]``.
    """

    @pytest.mark.skipif(
        any(name in os.environ for name in env),
        reason=f"{', '.join(env)} already set, so this is the rerun",
    )
    @pytest.mark.timeout(300, method="thread")
    def test(run_cmd):
        cmd = [
            sys.executable,
            "-m",
            "pytest",
            os.path.basename(test_file),
            *pytest_args,
        ]
        cwd = os.path.dirname(os.path.abspath(test_file))
        assert run_cmd(cmd=cmd, cwd=cwd, env=env) == 0

    return test
