# Copyright (c) 2021-2024, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.

# distutils: language = c++
# cython: language_level=3

from pathlib import Path

from libc.stdint cimport uintptr_t
from libcpp cimport bool as cpp_bool
from libcpp.utility cimport move

from kvikio._lib.arr cimport Array, mem_ptr_nbytes, parse_buffer_argument
from kvikio._lib.future cimport (
    IOFuture,
    IOFutureStream,
    _wrap_io_future,
    _wrap_stream_future,
)

from kvikio._lib import defaults


cdef class CuFile:
    def __init__(self, file_path, str flags="r"):
        self._handle = move(
            FileHandle(
                bytes(Path(file_path)),
                flags.encode(),
            )
        )

    cpdef close(self):
        self._handle.close()

    cpdef cpp_bool closed(self):
        return self._handle.closed()

    cpdef cpp_int fileno(self):
        return self._handle.fd()

    cpdef cpp_int open_flags(self):
        return self._handle.fd_open_flags()

    cpdef IOFuture pread(self, Array buf, Py_ssize_t size=-1, Py_ssize_t file_offset=0,
                         Py_ssize_t task_size=0):
        cdef mem_ptr_nbytes info = parse_buffer_argument(buf, size, True)
        return _wrap_io_future(
            self._handle.pread(
                <void*>info.ptr,
                info.nbytes,
                file_offset,
                task_size if task_size > 0 else defaults.task_size()
            )
        )

    cpdef IOFuture pwrite(self, Array buf, Py_ssize_t size=-1, Py_ssize_t file_offset=0,
                          Py_ssize_t task_size=0):
        cdef mem_ptr_nbytes info = parse_buffer_argument(buf, size, True)
        return _wrap_io_future(
            self._handle.pwrite(
                <void*>info.ptr,
                info.nbytes,
                file_offset,
                task_size if task_size > 0 else defaults.task_size()
            )
        )

    cpdef Py_ssize_t read(self, Array buf, Py_ssize_t size=-1, Py_ssize_t file_offset=0,
                          Py_ssize_t dev_offset=0):
        cdef mem_ptr_nbytes info = parse_buffer_argument(buf, size, False)
        return self._handle.read(
            <void*>info.ptr,
            info.nbytes,
            file_offset,
            dev_offset,
        )

    cpdef Py_ssize_t write(self, Array buf, Py_ssize_t size=-1,
                           Py_ssize_t file_offset=0, Py_ssize_t dev_offset=0):
        cdef mem_ptr_nbytes info = parse_buffer_argument(buf, size, False)
        return self._handle.write(
            <void*>info.ptr,
            info.nbytes,
            file_offset,
            dev_offset,
        )

    cpdef IOFutureStream read_async(self, Array buf, Py_ssize_t size=-1,
                                    Py_ssize_t file_offset=0, Py_ssize_t dev_offset=0,
                                    uintptr_t stream=0):
        cdef mem_ptr_nbytes info = parse_buffer_argument(buf, size, False)
        return _wrap_stream_future(self._handle.read_async(
            <void*>info.ptr,
            info.nbytes,
            file_offset,
            dev_offset,
            <CUstream>stream,
        ))

    cpdef IOFutureStream write_async(self, Array buf, Py_ssize_t size=-1,
                                     Py_ssize_t file_offset=0, Py_ssize_t dev_offset=0,
                                     uintptr_t stream=0):
        cdef mem_ptr_nbytes info = parse_buffer_argument(buf, size, False)
        return _wrap_stream_future(self._handle.write_async(
            <void*>info.ptr,
            info.nbytes,
            file_offset,
            dev_offset,
            <CUstream>stream,
        ))
