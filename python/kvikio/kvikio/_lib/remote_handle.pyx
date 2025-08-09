# Copyright (c) 2024-2025, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.

# distutils: language = c++
# cython: language_level=3

from typing import Optional

from cython.operator cimport dereference as deref
from libc.stdint cimport uintptr_t
from libcpp.memory cimport make_unique, unique_ptr
from libcpp.pair cimport pair
from libcpp.string cimport string
from libcpp.utility cimport move, pair

from kvikio._lib.arr cimport parse_buffer_argument
from kvikio._lib.future cimport IOFuture, _wrap_io_future, future


cdef extern from "<kvikio/remote_handle.hpp>" nogil:
    cdef cppclass cpp_RemoteEndpoint "kvikio::RemoteEndpoint":
        string str() except +

    cdef cppclass cpp_HttpEndpoint "kvikio::HttpEndpoint"(cpp_RemoteEndpoint):
        cpp_HttpEndpoint(string url) except +

    cdef cppclass cpp_S3Endpoint "kvikio::S3Endpoint"(cpp_RemoteEndpoint):
        cpp_S3Endpoint(string url) except +
        cpp_S3Endpoint(pair[string, string] bucket_and_object_names) except +

    pair[string, string] cpp_parse_s3_url \
        "kvikio::S3Endpoint::parse_s3_url"(string url) except +

    cdef cppclass cpp_S3EndpointWithPresignedUrl "kvikio::S3EndpointWithPresignedUrl" \
                                                 (cpp_RemoteEndpoint):
        cpp_S3EndpointWithPresignedUrl(string presigned_url) except +

    cdef cppclass cpp_RemoteHandle "kvikio::RemoteHandle":
        cpp_RemoteHandle(
            unique_ptr[cpp_RemoteEndpoint] endpoint, size_t nbytes
        ) except +
        cpp_RemoteHandle(unique_ptr[cpp_RemoteEndpoint] endpoint) except +
        size_t nbytes() except +
        const cpp_RemoteEndpoint& endpoint() except +
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


cdef string _to_string(str s):
    """Convert Python object to a C++ string (if None, return the empty string)"""
    if s is not None:
        return s.encode()
    else:
        return string()

cdef pair[string, string] _to_string_pair(str s1, str s2):
    """Wrap two Python string objects in a C++ pair"""
    return pair[string, string](_to_string(s1), _to_string(s2))


# Helper function to cast an endpoint to its base class `RemoteEndpoint`
cdef extern from * nogil:
    """
    template <typename T>
    std::unique_ptr<kvikio::RemoteEndpoint> cast_to_remote_endpoint(T endpoint)
    {
        return std::move(endpoint);
    }
    """
    cdef unique_ptr[cpp_RemoteEndpoint] cast_to_remote_endpoint[T](T handle) except +


cdef class RemoteFile:
    cdef unique_ptr[cpp_RemoteHandle] _handle

    @staticmethod
    cdef RemoteFile _from_endpoint(
        unique_ptr[cpp_RemoteEndpoint] ep,
        nbytes: Optional[int],
    ):
        cdef RemoteFile ret = RemoteFile()

        if nbytes is None:
            with nogil:
                ret._handle = make_unique[cpp_RemoteHandle](move(ep))
            return ret

        cdef size_t n = nbytes

        with nogil:
            ret._handle = make_unique[cpp_RemoteHandle](move(ep), n)
        return ret

    @staticmethod
    def open_http(
        url: str,
        nbytes: Optional[int],
    ):
        cdef string cpp_url = _to_string(url)
        cdef unique_ptr[cpp_RemoteEndpoint] cpp_endpoint

        with nogil:
            cpp_endpoint = cast_to_remote_endpoint(
                make_unique[cpp_HttpEndpoint](cpp_url)
            )

        return RemoteFile._from_endpoint(
            move(cpp_endpoint),
            nbytes
        )

    @staticmethod
    def open_s3(
        bucket_name: str,
        object_name: str,
        nbytes: Optional[int],
    ):
        cdef pair[string, string] bucket_and_object_names = _to_string_pair(
            bucket_name, object_name)
        cdef unique_ptr[cpp_RemoteEndpoint] cpp_endpoint

        with nogil:
            cpp_endpoint = cast_to_remote_endpoint(
                make_unique[cpp_S3Endpoint](bucket_and_object_names)
            )

        return RemoteFile._from_endpoint(
            move(cpp_endpoint),
            nbytes
        )

    @staticmethod
    def open_s3_from_http_url(
        url: str,
        nbytes: Optional[int],
    ):
        cdef string cpp_url = _to_string(url)
        cdef unique_ptr[cpp_RemoteEndpoint] cpp_endpoint

        with nogil:
            cpp_endpoint = cast_to_remote_endpoint(
                make_unique[cpp_S3Endpoint](cpp_url)
            )

        return RemoteFile._from_endpoint(
            move(cpp_endpoint),
            nbytes
        )

    @staticmethod
    def open_s3_from_s3_url(
        url: str,
        nbytes: Optional[int],
    ):
        cdef string cpp_url = _to_string(url)
        cdef pair[string, string] bucket_and_object_names
        cdef unique_ptr[cpp_RemoteEndpoint] cpp_endpoint

        with nogil:
            bucket_and_object_names = cpp_parse_s3_url(cpp_url)
            cpp_endpoint = cast_to_remote_endpoint(
                make_unique[cpp_S3Endpoint](bucket_and_object_names)
            )

        return RemoteFile._from_endpoint(
            move(cpp_endpoint),
            nbytes
        )

    @staticmethod
    def open_s3_from_http_presigned_url(
        presigned_url: str,
        nbytes: Optional[int],
    ):
        cdef string cpp_url = _to_string(presigned_url)
        cdef unique_ptr[cpp_RemoteEndpoint] cpp_endpoint

        with nogil:
            cpp_endpoint = cast_to_remote_endpoint(
                make_unique[cpp_S3EndpointWithPresignedUrl](cpp_url)
            )

        return RemoteFile._from_endpoint(
            move(cpp_endpoint),
            nbytes
        )

    def __str__(self) -> str:
        cdef string ep_str
        with nogil:
            ep_str = deref(self._handle).endpoint().str()
        return f'<{self.__class__.__name__} "{ep_str.decode()}">'

    def nbytes(self) -> int:
        cdef size_t result
        with nogil:
            result = deref(self._handle).nbytes()
        return result

    def read(self, buf, size: Optional[int], file_offset: int) -> int:
        cdef pair[uintptr_t, size_t] info = parse_buffer_argument(buf, size, True)
        cdef size_t cpp_file_offset = file_offset
        cdef size_t result

        with nogil:
            result = deref(self._handle).read(
                <void*>info.first,
                info.second,
                cpp_file_offset,
            )

        return result

    def pread(self, buf, size: Optional[int], file_offset: int) -> IOFuture:
        cdef pair[uintptr_t, size_t] info = parse_buffer_argument(buf, size, True)
        cdef size_t cpp_file_offset = file_offset
        cdef future[size_t] fut

        with nogil:
            fut = deref(self._handle).pread(
                <void*>info.first,
                info.second,
                cpp_file_offset,
            )

        return _wrap_io_future(fut)
