# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.

# distutils: language = c++
# cython: language_level=3

from typing import Optional

from libc.stdint cimport uintptr_t
from libcpp.string cimport string
from libcpp.utility cimport pair

from kvikio._lib.arr cimport parse_buffer_argument
from kvikio._lib.future cimport IOFuture, _wrap_io_future, future


cdef extern from "<kvikio/remote_handle.hpp>" namespace "kvikio" nogil:
    cdef cppclass RemoteHandle:
        RemoteHandle() except +
        RemoteHandle(
            string bucket_name,
            string object_name,
        ) except +
        RemoteHandle(
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


cdef class RemoteFile:
    """ Remote file handle"""
    cdef RemoteHandle _handle

    @classmethod
    def from_bucket_and_object(cls, bucket_name: str, object_name: str):
        cdef RemoteFile ret = RemoteFile()
        ret._handle = RemoteHandle(
            str.encode(str(bucket_name)),
            str.encode(str(object_name)),
        )
        return ret

    @classmethod
    def from_url(cls, url: str):
        cdef RemoteFile ret = RemoteFile()
        ret._handle = RemoteHandle(str.encode(str(url)))
        return ret

    def nbytes(self) -> int:
        return self._handle.nbytes()

    def read(self, buf, size: Optional[int], file_offset: int) -> int:
        cdef pair[uintptr_t, size_t] info = parse_buffer_argument(buf, size, True)
        return self._handle.read(
            <void*>info.first,
            info.second,
            file_offset,
        )

    def pread(self, buf, size: Optional[int], file_offset: int) -> IOFuture:
        cdef pair[uintptr_t, size_t] info = parse_buffer_argument(buf, size, True)
        return _wrap_io_future(
            self._handle.pread(
                <void*>info.first,
                info.second,
                file_offset,
            )
        )
