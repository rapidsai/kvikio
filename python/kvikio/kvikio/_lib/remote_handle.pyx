# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.

# distutils: language = c++
# cython: language_level=3

from typing import Optional

from cython.operator cimport dereference as deref
from libc.stdint cimport uintptr_t
from libcpp.memory cimport make_shared, make_unique, shared_ptr, unique_ptr
from libcpp.string cimport string
from libcpp.utility cimport pair

from kvikio._lib.arr cimport parse_buffer_argument
from kvikio._lib.future cimport IOFuture, _wrap_io_future, future


cdef extern from "<kvikio/remote_handle.hpp>" nogil:

    cdef cppclass cpp_S3Context "kvikio::S3Context":
        cpp_S3Context() except +
        cpp_S3Context(string endpoint_override) except +

    cdef cppclass cpp_RemoteHandle "kvikio::RemoteHandle":
        cpp_RemoteHandle(
            shared_ptr[cpp_S3Context] context,
            string bucket_name,
            string object_name,
        ) except +
        cpp_RemoteHandle(
            shared_ptr[cpp_S3Context] context,
            string remote_path,
        ) except +
        int nbytes()
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

cdef class S3Context:
    cdef shared_ptr[cpp_S3Context] _handle

    def __init__(self, endpoint_override: Optional[str]):
        if endpoint_override is None:
            self._handle = make_shared[cpp_S3Context]()
            return
        cdef string s = str.encode(str(endpoint_override))
        self._handle = make_shared[cpp_S3Context](s)

cdef class RemoteFile:
    cdef unique_ptr[cpp_RemoteHandle] _handle

    @classmethod
    def from_bucket_and_object(
        cls,
        S3Context context,
        bucket_name: str,
        object_name: str
    ):
        cdef RemoteFile ret = RemoteFile()
        cdef string b = str.encode(str(bucket_name))
        cdef string o = str.encode(str(object_name))
        ret._handle = make_unique[cpp_RemoteHandle](context._handle, b, o)
        return ret

    @classmethod
    def from_url(cls, S3Context context, url: str):
        cdef RemoteFile ret = RemoteFile()
        cdef string u = str.encode(str(url))
        ret._handle = make_unique[cpp_RemoteHandle](context._handle, u)
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
