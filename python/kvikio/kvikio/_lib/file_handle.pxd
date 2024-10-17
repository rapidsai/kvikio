# Copyright (c) 2021-2024, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.

# distutils: language = c++
# cython: language_level=3

from posix cimport fcntl

from libc.stdint cimport uintptr_t
from libcpp cimport bool as cpp_bool
from libcpp.string cimport string

from kvikio._lib import defaults
from kvikio._lib.arr cimport Array
from kvikio._lib.future cimport IOFuture, IOFutureStream, cpp_StreamFuture, future

ctypedef int cpp_int


cdef extern from "cuda.h":
    ctypedef void* CUstream


cdef extern from "<kvikio/file_handle.hpp>" namespace "kvikio" nogil:
    cdef cppclass FileHandle:
        FileHandle() except +
        FileHandle(cpp_int fd) except +
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
        cpp_bool closed()
        cpp_int fd()
        cpp_int fd_open_flags() except +
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

    cpdef close(self)
    cpdef cpp_bool closed(self)
    cpdef cpp_int fileno(self)
    cpdef cpp_int open_flags(self)
    cpdef IOFuture pread(self, Array buf, Py_ssize_t size=*, Py_ssize_t file_offset=*,
                         Py_ssize_t task_size=*)
    cpdef IOFuture pwrite(self, Array buf, Py_ssize_t size=*, Py_ssize_t file_offset=*,
                          Py_ssize_t task_size=*)
    cpdef Py_ssize_t read(self, Array buf, Py_ssize_t size=*, Py_ssize_t file_offset=*,
                          Py_ssize_t dev_offset=*)
    cpdef Py_ssize_t write(self, Array buf, Py_ssize_t size=*, Py_ssize_t file_offset=*,
                           Py_ssize_t dev_offset=*)
    cpdef IOFutureStream read_async(self, Array buf, Py_ssize_t size=*,
                                    Py_ssize_t file_offset=*, Py_ssize_t dev_offset=*,
                                    uintptr_t stream=*)
    cpdef IOFutureStream write_async(self, Array buf, Py_ssize_t size=*,
                                     Py_ssize_t file_offset=*, Py_ssize_t dev_offset=*,
                                     uintptr_t stream=*)
