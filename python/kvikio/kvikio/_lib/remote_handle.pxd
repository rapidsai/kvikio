# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.

# distutils: language = c++
# cython: language_level=3

from libcpp.memory cimport unique_ptr
from libcpp.string cimport string

from kvikio._lib.arr cimport Array
from kvikio._lib.future cimport IOFuture, future


cdef extern from "<kvikio/remote_handle.hpp>" nogil:
    cdef cppclass cpp_RemoteEndpoint "kvikio::RemoteEndpoint":
        pass

    cdef cppclass cpp_HttpEndpoint "kvikio::HttpEndpoint":
        cpp_HttpEndpoint(string url) except +

    cdef cppclass cpp_RemoteHandle "kvikio::RemoteHandle":
        cpp_RemoteHandle(
            unique_ptr[cpp_RemoteEndpoint] endpoint, size_t nbytes
        ) except +
        cpp_RemoteHandle(unique_ptr[cpp_RemoteEndpoint] endpoint) except +
        size_t nbytes() except +
        size_t read(
            void* buf,
            size_t size,
            size_t file_offset
        ) except +
        future[size_t] pread(
            void* buf,
            size_t size,
            size_t file_offset
        ) except +


cdef class RemoteFile:
    cdef unique_ptr[cpp_RemoteHandle] _handle

    @staticmethod
    cdef RemoteFile cpp_open_http(str url, Py_ssize_t nbytes=*)

    cpdef Py_ssize_t nbytes(self)

    cpdef Py_ssize_t read(self, Array buf, Py_ssize_t size=*, Py_ssize_t file_offset=*)
    cpdef IOFuture pread(self, Array buf, Py_ssize_t size=*, Py_ssize_t file_offset=*)
