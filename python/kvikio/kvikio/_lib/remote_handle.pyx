# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.

# distutils: language = c++
# cython: language_level=3

from typing import Optional

from cython.operator cimport dereference as deref
from libc.stdint cimport uintptr_t
from libcpp.memory cimport make_unique, unique_ptr
from libcpp.string cimport string
from libcpp.utility cimport pair

from kvikio._lib.arr cimport parse_buffer_argument
from kvikio._lib.future cimport IOFuture, _wrap_io_future, future


cdef extern from "<kvikio/remote_handle.hpp>" nogil:
    cdef cppclass cpp_RemoteHandle "kvikio::RemoteHandle":
        cpp_RemoteHandle(string url, size_t nbytes) except +
        cpp_RemoteHandle(string url) except +
        int nbytes() except +
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

    def read(self, buf, size: Optional[int], file_offset: int) -> int:
        cdef pair[uintptr_t, size_t] info = parse_buffer_argument(buf, size, True)
        return deref(self._handle).read(
            <void*>info.first,
            info.second,
            file_offset,
        )

    def pread(self, buf, size: Optional[int], file_offset: int) -> IOFuture:
        cdef pair[uintptr_t, size_t] info = parse_buffer_argument(buf, size, True)
        return _wrap_io_future(
            deref(self._handle).pread(
                <void*>info.first,
                info.second,
                file_offset,
            )
        )
