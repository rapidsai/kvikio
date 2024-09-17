# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.

# distutils: language = c++
# cython: language_level=3

from typing import Optional

from cython.operator cimport dereference as deref
from libcpp.memory cimport make_unique, unique_ptr
from libcpp.string cimport string

# from kvikio._lib.arr cimport parse_buffer_argument
# from kvikio._lib.future cimport IOFuture, _wrap_io_future, future


cdef extern from "<kvikio/remote_handle.hpp>" nogil:
    cdef cppclass cpp_RemoteHandle "kvikio::RemoteHandle":
        cpp_RemoteHandle(string url, size_t nbytes) except +
        cpp_RemoteHandle(string url) except +
        int nbytes() except +


cdef class RemoteFile:
    cdef unique_ptr[cpp_RemoteHandle] _handle

    @classmethod
    def from_url(cls, url: str, nbytes: Optional[int]):
        cdef RemoteFile ret = RemoteFile()
        cdef string u = str.encode(str(url))
        if nbytes is None:
            ret._handle = make_unique[cpp_RemoteHandle](u)
            return ret
        cdef size_t n = nbytes
        ret._handle = make_unique[cpp_RemoteHandle](u, n)
        return ret

    def nbytes(self) -> int:
        return deref(self._handle).nbytes()
