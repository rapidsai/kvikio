# SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# distutils: language = c++
# cython: language_level=3

import io
import os
from typing import Optional, Union

from posix cimport fcntl

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


cdef extern from "cuda.h":
    ctypedef void* CUstream


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
        cdef string cpp_file_path = os.fsencode(file_path)
        cdef string cpp_flags = str(flags).encode()
        with nogil:
            self._handle = move(
                FileHandle(
                    cpp_file_path,
                    cpp_flags
                )
            )

    def close(self) -> None:
        with nogil:
            self._handle.close()

    def closed(self) -> bool:
        cdef bool result
        with nogil:
            result = self._handle.closed()
        return result

    def fileno(self) -> int:
        cdef int result
        with nogil:
            result = self._handle.fd()
        return result

    def open_flags(self) -> int:
        cdef int result
        with nogil:
            result = self._handle.fd_open_flags()
        return result

    def pread(self, buf, size: Optional[int], file_offset: int, task_size) -> IOFuture:
        cdef pair[uintptr_t, size_t] info = parse_buffer_argument(buf, size, True)
        cdef size_t cpp_file_offset = file_offset
        cdef size_t cpp_task_size = task_size if task_size else defaults.task_size()
        cdef future[size_t] fut
        with nogil:
            fut = self._handle.pread(
                <void*>info.first,
                info.second,
                cpp_file_offset,
                cpp_task_size
            )
        return _wrap_io_future(fut)

    def pwrite(self, buf, size: Optional[int], file_offset: int, task_size) -> IOFuture:
        cdef pair[uintptr_t, size_t] info = parse_buffer_argument(buf, size, True)
        cdef size_t cpp_file_offset = file_offset
        cdef size_t cpp_task_size = task_size if task_size else defaults.task_size()
        cdef future[size_t] fut
        with nogil:
            fut = self._handle.pwrite(
                <void*>info.first,
                info.second,
                cpp_file_offset,
                cpp_task_size
            )
        return _wrap_io_future(fut)

    def read(self, buf, size: Optional[int], file_offset: int, dev_offset: int) -> int:
        cdef pair[uintptr_t, size_t] info = parse_buffer_argument(buf, size, False)
        cdef size_t cpp_file_offset = file_offset
        cdef size_t cpp_dev_offset = dev_offset
        cdef size_t result
        with nogil:
            result = self._handle.read(
                <void*>info.first,
                info.second,
                cpp_file_offset,
                cpp_dev_offset,
            )
        return result

    def write(self, buf, size: Optional[int], file_offset: int, dev_offset: int) -> int:
        cdef pair[uintptr_t, size_t] info = parse_buffer_argument(buf, size, False)
        cdef size_t cpp_file_offset = file_offset
        cdef size_t cpp_dev_offset = dev_offset
        cdef size_t result
        with nogil:
            result = self._handle.write(
                <void*>info.first,
                info.second,
                cpp_file_offset,
                cpp_dev_offset,
            )
        return result

    def read_async(self, buf, size: Optional[int], file_offset: int, dev_offset: int,
                   st: uintptr_t) -> IOFutureStream:
        cdef CUstream stream = <CUstream>st
        cdef pair[uintptr_t, size_t] info = parse_buffer_argument(buf, size, False)
        cdef size_t cpp_file_offset = file_offset
        cdef size_t cpp_dev_offset = dev_offset
        cdef cpp_StreamFuture fut
        with nogil:
            fut = self._handle.read_async(
                <void*>info.first,
                info.second,
                cpp_file_offset,
                cpp_dev_offset,
                stream,
            )
        return _wrap_stream_future(fut)

    def write_async(self, buf, size: Optional[int], file_offset: int, dev_offset: int,
                    st: uintptr_t) -> IOFutureStream:
        cdef CUstream stream = <CUstream>st
        cdef pair[uintptr_t, size_t] info = parse_buffer_argument(buf, size, False)
        cdef size_t cpp_file_offset = file_offset
        cdef size_t cpp_dev_offset = dev_offset
        cdef cpp_StreamFuture fut
        with nogil:
            fut = self._handle.write_async(
                <void*>info.first,
                info.second,
                cpp_file_offset,
                cpp_dev_offset,
                stream,
            )
        return _wrap_stream_future(fut)

cdef extern from "<kvikio/file_utils.hpp>" nogil:
    pair[size_t, size_t] cpp_get_page_cache_info_str \
        "kvikio::get_page_cache_info"(string file_path) except +

    pair[size_t, size_t] cpp_get_page_cache_info_int \
        "kvikio::get_page_cache_info"(int fd) except +

    bool cpp_clear_page_cache "kvikio::clear_page_cache" \
        (bool reclaim_dentries_and_inodes, bool clear_dirty_pages) \
        except +


def get_page_cache_info(file: Union[os.PathLike, str, int, io.IOBase]) \
        -> tuple[int, int]:
    cdef pair[size_t, size_t] result
    cdef string path_bytes
    cdef int fd

    if isinstance(file, os.PathLike) or isinstance(file, str):
        # file is a path or a string object
        path_bytes = os.fsencode(file)
        with nogil:
            result = cpp_get_page_cache_info_str(path_bytes)
        return result
    elif isinstance(file, int):
        # file is a file descriptor
        fd = file
        with nogil:
            result = cpp_get_page_cache_info_int(fd)
        return result
    elif isinstance(file, io.IOBase):
        # file is a file object
        # pass its file descriptor to the underlying C++ function
        fd = file.fileno()
        with nogil:
            result = cpp_get_page_cache_info_int(fd)
        return result
    else:
        raise ValueError("The type of `file` must be `os.PathLike`, `str`, `int`, "
                         "or `io.IOBase`")


def clear_page_cache(reclaim_dentries_and_inodes: bool,
                     clear_dirty_pages: bool) -> bool:
    cdef bool result
    with nogil:
        result = cpp_clear_page_cache(reclaim_dentries_and_inodes, clear_dirty_pages)
    return result
