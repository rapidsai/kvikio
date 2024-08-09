# Copyright (c) 2021-2024, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.

# distutils: language = c++
# cython: language_level=3

from posix cimport fcntl

from libcpp cimport bool
from libcpp.string cimport string
from libcpp.vector cimport vector

from kvikio._lib.future cimport cpp_StreamFuture, future


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
