# Copyright (c) 2021-2024, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.

# distutils: language = c++
# cython: language_level=3

import pathlib
from typing import Optional

from libc.stdint cimport uintptr_t
from libcpp.utility cimport move, pair

from kvikio._lib.future cimport (
    IOFuture,
    IOFutureStream,
    _wrap_io_future,
    _wrap_stream_future,
)
from kvikio._lib import defaults
from kvikio._lib.arr cimport Array

from .kvikio_cxx_api cimport CUstream, FileHandle


cdef pair[uintptr_t, size_t] _parse_buffer(buf, size, bint accept_host_buffer) except *:
    """Parse `buf` and `size` argument and return a pointer and nbytes"""
    if not isinstance(buf, Array):
        buf = Array(buf)
    cdef Array arr = buf
    if not arr._contiguous():
        raise ValueError("Array must be contiguous")
    if not accept_host_buffer and not arr.cuda:
        raise ValueError("Non-CUDA buffers not supported")
    cdef size_t nbytes
    if size is None:
        nbytes = arr.nbytes
    elif size > arr.nbytes:
        raise ValueError("Size is greater than the size of the buffer")
    else:
        nbytes = size
    return pair[uintptr_t, size_t](arr.ptr, nbytes)


cdef class CuFile:
    """ File handle for GPUDirect Storage (GDS) """
    cdef FileHandle _handle

    def __init__(self, file_path, flags="r"):
        self._handle = move(
            FileHandle(
                str.encode(str(pathlib.Path(file_path))),
                str.encode(str(flags))
            )
        )

    def close(self) -> None:
        self._handle.close()

    def closed(self) -> bool:
        return self._handle.closed()

    def fileno(self) -> int:
        return self._handle.fd()

    def open_flags(self) -> int:
        return self._handle.fd_open_flags()

    def pread(self, buf, size: Optional[int], file_offset: int, task_size) -> IOFuture:
        cdef pair[uintptr_t, size_t] info = _parse_buffer(buf, size, True)
        return _wrap_io_future(
            self._handle.pread(
                <void*>info.first,
                info.second,
                file_offset,
                task_size if task_size else defaults.task_size()
            )
        )

    def pwrite(self, buf, size: Optional[int], file_offset: int, task_size) -> IOFuture:
        cdef pair[uintptr_t, size_t] info = _parse_buffer(buf, size, True)
        return _wrap_io_future(
            self._handle.pwrite(
                <void*>info.first,
                info.second,
                file_offset,
                task_size if task_size else defaults.task_size()
            )
        )

    def read(self, buf, size: Optional[int], file_offset: int, dev_offset: int) -> int:
        cdef pair[uintptr_t, size_t] info = _parse_buffer(buf, size, False)
        return self._handle.read(
            <void*>info.first,
            info.second,
            file_offset,
            dev_offset,
        )

    def write(self, buf, size: Optional[int], file_offset: int, dev_offset: int) -> int:
        cdef pair[uintptr_t, size_t] info = _parse_buffer(buf, size, False)
        return self._handle.write(
            <void*>info.first,
            info.second,
            file_offset,
            dev_offset,
        )

    def read_async(self, buf, size: Optional[int], file_offset: int, dev_offset: int,
                   st: uintptr_t) -> IOFutureStream:
        stream = <CUstream>st
        cdef pair[uintptr_t, size_t] info = _parse_buffer(buf, size, False)
        return _wrap_stream_future(self._handle.read_async(
            <void*>info.first,
            info.second,
            file_offset,
            dev_offset,
            stream,
        ))

    def write_async(self, buf, size: Optional[int], file_offset: int, dev_offset: int,
                    st: uintptr_t) -> IOFutureStream:
        stream = <CUstream>st
        cdef pair[uintptr_t, size_t] info = _parse_buffer(buf, size, False)
        return _wrap_stream_future(self._handle.write_async(
            <void*>info.first,
            info.second,
            file_offset,
            dev_offset,
            stream,
        ))
