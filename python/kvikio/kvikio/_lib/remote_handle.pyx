# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.

# distutils: language = c++
# cython: language_level=3

from cython cimport Py_ssize_t
from cython.operator cimport dereference as deref
from libcpp.memory cimport make_unique, unique_ptr
from libcpp.string cimport string
from libcpp.utility cimport move

from kvikio._lib.arr cimport Array, mem_ptr_nbytes, parse_buffer_argument
from kvikio._lib.future cimport IOFuture, _wrap_io_future


cdef string _to_string(str s):
    """Convert Python object to a C++ string (if None, return the empty string)"""
    if s is not None:
        return s.encode()
    else:
        return string()


cdef class RemoteFile:
    @staticmethod
    def open_http(str url, Py_ssize_t nbytes=-1) -> RemoteFile:
        return RemoteFile.cpp_open_http(url, nbytes)

    @staticmethod
    cdef RemoteFile cpp_open_http(str url, Py_ssize_t nbytes=-1):
        cdef RemoteFile ret = RemoteFile()
        cdef unique_ptr[cpp_HttpEndpoint] ep = make_unique[cpp_HttpEndpoint](
            _to_string(url)
        )
        if nbytes < 0:
            ret._handle = make_unique[cpp_RemoteHandle](move(ep))
            return ret
        ret._handle = make_unique[cpp_RemoteHandle](move(ep), nbytes)
        return ret

    cpdef Py_ssize_t nbytes(self):
        return deref(self._handle).nbytes()

    cpdef Py_ssize_t read(self, Array buf, Py_ssize_t size=-1,
                          Py_ssize_t file_offset=0):
        cdef mem_ptr_nbytes info = parse_buffer_argument(buf, size, True)
        return deref(self._handle).read(
            <void*>info.ptr,
            info.nbytes,
            file_offset,
        )

    cpdef IOFuture pread(self, Array buf, Py_ssize_t size=-1, Py_ssize_t file_offset=0):
        cdef mem_ptr_nbytes info = parse_buffer_argument(buf, size, True)
        return _wrap_io_future(
            deref(self._handle).pread(
                <void*>info.ptr,
                info.nbytes,
                file_offset,
            )
        )
