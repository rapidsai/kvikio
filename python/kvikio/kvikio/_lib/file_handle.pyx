# Copyright (c) 2021-2024, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.

# distutils: language = c++
# cython: language_level=3

import pathlib
from typing import Optional

from posix cimport fcntl

from cuda.ccuda cimport CUstream
from libc.stdint cimport uintptr_t
from libcpp cimport bool
from libcpp.string cimport string
from libcpp.utility cimport move, pair

from kvikio._lib.arr cimport parse_buffer_argument
from kvikio._lib.future cimport (
    IOFuture,
    IOFutureStream,
    _wrap_io_future,
    _wrap_stream_future,
    cpp_StreamFuture,
    future,
)

from kvikio._lib import defaults


cdef extern from "<kvikio/file_handle.hpp>" namespace "kvikio" nogil:
    cdef cppclass FileHandle:
        FileHandle() except +
        FileHandle(int fd) except +
        FileHandle(
            string file_path,
            string flags,
        ) except +
        FileHandle(
            string file_path,
            string flags,
            fcntl.mode_t mode
        ) except +
        void close()
        bool closed()
        int fd()
        int fd_open_flags() except +
        future[size_t] pread(
            void* devPtr,
            size_t size,
            size_t file_offset,
            size_t task_size
        ) except +
        future[size_t] pwrite(
            void* devPtr,
            size_t size,
            size_t file_offset,
            size_t task_size
        ) except +
        size_t read(
            void* devPtr_base,
            size_t size,
            size_t file_offset,
            size_t devPtr_offset
        ) except +
        size_t write(
            void* devPtr_base,
            size_t size,
            size_t file_offset,
            size_t devPtr_offset
        ) except +
        cpp_StreamFuture read_async(
            void* devPtr_base,
            size_t size,
            size_t file_offset,
            size_t devPtr_offset,
            CUstream stream
        ) except +
        cpp_StreamFuture write_async(
            void* devPtr_base,
            size_t size,
            size_t file_offset,
            size_t devPtr_offset,
            CUstream stream
        ) except +


cdef class CuFile:
    """File handle for GPUDirect Storage (GDS)"""
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
        cdef pair[uintptr_t, size_t] info = parse_buffer_argument(buf, size, True)
        return _wrap_io_future(
            self._handle.pread(
                <void*>info.first,
                info.second,
                file_offset,
                task_size if task_size else defaults.task_size()
            )
        )

    def pwrite(self, buf, size: Optional[int], file_offset: int, task_size) -> IOFuture:
        cdef pair[uintptr_t, size_t] info = parse_buffer_argument(buf, size, True)
        return _wrap_io_future(
            self._handle.pwrite(
                <void*>info.first,
                info.second,
                file_offset,
                task_size if task_size else defaults.task_size()
            )
        )

    def read(self, buf, size: Optional[int], file_offset: int, dev_offset: int) -> int:
        cdef pair[uintptr_t, size_t] info = parse_buffer_argument(buf, size, False)
        return self._handle.read(
            <void*>info.first,
            info.second,
            file_offset,
            dev_offset,
        )

    def write(self, buf, size: Optional[int], file_offset: int, dev_offset: int) -> int:
        cdef pair[uintptr_t, size_t] info = parse_buffer_argument(buf, size, False)
        return self._handle.write(
            <void*>info.first,
            info.second,
            file_offset,
            dev_offset,
        )

    def read_async(self, buf, size: Optional[int], file_offset: int, dev_offset: int,
                   st: uintptr_t) -> IOFutureStream:
        stream = <CUstream>st
        cdef pair[uintptr_t, size_t] info = parse_buffer_argument(buf, size, False)
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
        cdef pair[uintptr_t, size_t] info = parse_buffer_argument(buf, size, False)
        return _wrap_stream_future(self._handle.write_async(
            <void*>info.first,
            info.second,
            file_offset,
            dev_offset,
            stream,
        ))
