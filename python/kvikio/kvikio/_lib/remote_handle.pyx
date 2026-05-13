# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# distutils: language = c++
# cython: language_level=3

from typing import Optional

from cython.operator cimport dereference as deref
from libc.stdint cimport uint8_t, uintptr_t
from libcpp.memory cimport make_unique, unique_ptr
from libcpp.optional cimport nullopt, optional
from libcpp.pair cimport pair
from libcpp.string cimport string
from libcpp.utility cimport move, pair
from libcpp.vector cimport vector

from kvikio._lib.arr cimport parse_buffer_argument
from kvikio._lib.future cimport IOFuture, _wrap_io_future, future


cdef extern from "<kvikio/remote_handle.hpp>" namespace "kvikio" nogil:
    cpdef enum class RemoteEndpointType(uint8_t):
        AUTO = 0
        S3 = 1
        S3_PUBLIC = 2
        S3_PRESIGNED_URL = 3
        WEBHDFS = 4
        HTTP = 5
    cdef cppclass cpp_RemoteEndpoint "kvikio::RemoteEndpoint":
        string str() except +

    cdef cppclass cpp_HttpEndpoint "kvikio::HttpEndpoint"(cpp_RemoteEndpoint):
        cpp_HttpEndpoint(string url) except +

    cdef cppclass cpp_S3Endpoint "kvikio::S3Endpoint"(cpp_RemoteEndpoint):
        cpp_S3Endpoint(
            string url,
            optional[string] aws_region,
            optional[string] aws_access_key,
            optional[string] aws_secret_access_key,
            optional[string] aws_session_token
        ) except +
        cpp_S3Endpoint(
            pair[string, string] bucket_and_object_names,
            optional[string] aws_region,
            optional[string] aws_access_key,
            optional[string] aws_secret_access_key,
            optional[string] aws_endpoint_url,
            optional[string] aws_session_token
        ) except +

    pair[string, string] cpp_parse_s3_url \
        "kvikio::S3Endpoint::parse_s3_url"(string url) except +

    cdef cppclass cpp_S3PublicEndpoint "kvikio::S3PublicEndpoint" (cpp_RemoteEndpoint):
        cpp_S3PublicEndpoint(string url) except +

    cdef cppclass cpp_S3EndpointWithPresignedUrl "kvikio::S3EndpointWithPresignedUrl" \
                                                 (cpp_RemoteEndpoint):
        cpp_S3EndpointWithPresignedUrl(string presigned_url) except +

    cdef cppclass cpp_RemoteHandle "kvikio::RemoteHandle":
        cpp_RemoteHandle(
            unique_ptr[cpp_RemoteEndpoint] endpoint, size_t nbytes
        ) except +
        cpp_RemoteHandle(unique_ptr[cpp_RemoteEndpoint] endpoint) except +
        RemoteEndpointType remote_endpoint_type() noexcept
        size_t nbytes() noexcept
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

        @staticmethod
        cpp_RemoteHandle cpp_easy_open "open"(
            string url,
            RemoteEndpointType remote_endpoint_type,
            optional[vector[RemoteEndpointType]] allow_list,
            optional[size_t] nbytes
        ) except +

cdef extern from "<kvikio/hdfs.hpp>" nogil:
    cdef cppclass cpp_WebHdfsEndpoint "kvikio::WebHdfsEndpoint"(cpp_RemoteEndpoint):
        cpp_WebHdfsEndpoint(string url) except +

cdef string _to_string(str s):
    """Convert Python object to a C++ string (if None, return the empty string)"""
    if s is not None:
        return s.encode()
    else:
        return string()

cdef pair[string, string] _to_string_pair(str s1, str s2):
    """Wrap two Python string objects in a C++ pair"""
    return pair[string, string](_to_string(s1), _to_string(s2))

cdef optional[string] _to_optional_string(str s):
    """Convert Python object to a C++ optional string (if None, return nullopt)"""
    cdef optional[string] result
    if s is None:
        result = nullopt
    else:
        result = optional[string](_to_string(s))
    return result


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

# Helper function for the cpp_RemoteHandle.open method to return
# unique_ptr[cpp_RemoteHandle] instead of cpp_RemoteHandle. Due to lack of a nullary
# constructor, cpp_RemoteHandle cannot be created as a stack variable in Cython.
cdef extern from * nogil:
    """
    inline std::unique_ptr<kvikio::RemoteHandle> create_remote_handle_from_open(
        std::string url,
        kvikio::RemoteEndpointType remote_endpoint_type,
        std::optional<std::vector<kvikio::RemoteEndpointType>> allow_list,
        std::optional<std::size_t> nbytes)
    {
        return std::make_unique<kvikio::RemoteHandle>(
            kvikio::RemoteHandle::open(url, remote_endpoint_type, allow_list, nbytes)
        );
    }
    """
    cdef unique_ptr[cpp_RemoteHandle] create_remote_handle_from_open(
        string url,
        RemoteEndpointType remote_endpoint_type,
        optional[vector[RemoteEndpointType]] allow_list,
        optional[size_t] nbytes
    ) except +

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
        aws_region_name: Optional[str] = None,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        aws_endpoint_url: Optional[str] = None,
        aws_session_token: Optional[str] = None,
    ):
        cdef pair[string, string] bucket_and_object_names = _to_string_pair(
            bucket_name, object_name
        )
        cdef optional[string] cpp_aws_region = _to_optional_string(aws_region_name)
        cdef optional[string] cpp_aws_access_key = _to_optional_string(
            aws_access_key_id
        )
        cdef optional[string] cpp_aws_secret_access_key = (
            _to_optional_string(aws_secret_access_key)
        )
        cdef optional[string] cpp_aws_endpoint_url = _to_optional_string(
            aws_endpoint_url
        )
        cdef optional[string] cpp_aws_session_token = _to_optional_string(
            aws_session_token
        )
        cdef unique_ptr[cpp_RemoteEndpoint] cpp_endpoint

        with nogil:
            cpp_endpoint = cast_to_remote_endpoint(
                make_unique[cpp_S3Endpoint](
                    bucket_and_object_names,
                    cpp_aws_region,
                    cpp_aws_access_key,
                    cpp_aws_secret_access_key,
                    cpp_aws_endpoint_url,
                    cpp_aws_session_token
                )
            )

        return RemoteFile._from_endpoint(
            move(cpp_endpoint),
            nbytes
        )

    @staticmethod
    def open_s3_from_http_url(
        url: str,
        nbytes: Optional[int],
        aws_region_name: Optional[str] = None,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        aws_session_token: Optional[str] = None,
    ):
        cdef string cpp_url = _to_string(url)
        cdef optional[string] cpp_aws_region = _to_optional_string(aws_region_name)
        cdef optional[string] cpp_aws_access_key = _to_optional_string(
            aws_access_key_id
        )
        cdef optional[string] cpp_aws_secret_access_key = (
            _to_optional_string(aws_secret_access_key)
        )
        cdef optional[string] cpp_aws_session_token = _to_optional_string(
            aws_session_token
        )
        cdef unique_ptr[cpp_RemoteEndpoint] cpp_endpoint

        with nogil:
            cpp_endpoint = cast_to_remote_endpoint(
                make_unique[cpp_S3Endpoint](
                    cpp_url,
                    cpp_aws_region,
                    cpp_aws_access_key,
                    cpp_aws_secret_access_key,
                    cpp_aws_session_token
                )
            )

        return RemoteFile._from_endpoint(
            move(cpp_endpoint),
            nbytes
        )

    @staticmethod
    def open_s3_from_s3_url(
        url: str,
        nbytes: Optional[int],
        aws_region_name: Optional[str] = None,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        aws_endpoint_url: Optional[str] = None,
        aws_session_token: Optional[str] = None,
    ):
        cdef string cpp_url = _to_string(url)
        cdef pair[string, string] bucket_and_object_names
        cdef optional[string] cpp_aws_region = _to_optional_string(aws_region_name)
        cdef optional[string] cpp_aws_access_key = _to_optional_string(
            aws_access_key_id
        )
        cdef optional[string] cpp_aws_secret_access_key = (
            _to_optional_string(aws_secret_access_key)
        )
        cdef optional[string] cpp_aws_endpoint_url = _to_optional_string(
            aws_endpoint_url
        )
        cdef optional[string] cpp_aws_session_token = _to_optional_string(
            aws_session_token
        )
        cdef unique_ptr[cpp_RemoteEndpoint] cpp_endpoint

        with nogil:
            bucket_and_object_names = cpp_parse_s3_url(cpp_url)
            cpp_endpoint = cast_to_remote_endpoint(
                make_unique[cpp_S3Endpoint](
                    bucket_and_object_names,
                    cpp_aws_region,
                    cpp_aws_access_key,
                    cpp_aws_secret_access_key,
                    cpp_aws_endpoint_url,
                    cpp_aws_session_token
                )
            )

        return RemoteFile._from_endpoint(
            move(cpp_endpoint),
            nbytes
        )

    @staticmethod
    def open_s3_public(
        url: str,
        nbytes: Optional[int],
    ):
        cdef string cpp_url = _to_string(url)
        cdef unique_ptr[cpp_RemoteEndpoint] cpp_endpoint

        with nogil:
            cpp_endpoint = cast_to_remote_endpoint(
                make_unique[cpp_S3PublicEndpoint](cpp_url)
            )

        return RemoteFile._from_endpoint(
            move(cpp_endpoint),
            nbytes
        )

    @staticmethod
    def open_s3_presigned_url(
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

    @staticmethod
    def open_webhdfs(
        url: str,
        nbytes: Optional[int],
    ):
        return RemoteFile._from_endpoint(
            cast_to_remote_endpoint(
                make_unique[cpp_WebHdfsEndpoint](_to_string(url))
            ),
            nbytes
        )

    @staticmethod
    def open(
        url: str,
        remote_endpoint_type: RemoteEndpointType,
        allow_list: Optional[list],
        nbytes: Optional[int]
    ):
        cdef optional[vector[RemoteEndpointType]] cpp_allow_list
        cdef vector[RemoteEndpointType] vec_allow_list
        if allow_list is None:
            cpp_allow_list = nullopt
        else:
            for allow_item in allow_list:
                vec_allow_list.push_back(allow_item.value)
            cpp_allow_list = vec_allow_list

        cdef optional[size_t] cpp_nbytes
        if nbytes is None:
            cpp_nbytes = nullopt
        else:
            cpp_nbytes = <size_t>nbytes

        cdef RemoteFile ret = RemoteFile()
        cdef unique_ptr[cpp_RemoteHandle] cpp_handle
        cdef string cpp_url = _to_string(url)
        with nogil:
            cpp_handle = create_remote_handle_from_open(
                cpp_url,
                remote_endpoint_type,
                cpp_allow_list,
                cpp_nbytes)
        ret._handle = move(cpp_handle)

        return ret

    def __str__(self) -> str:
        cdef string ep_str
        with nogil:
            ep_str = deref(self._handle).endpoint().str()
        return f'<{self.__class__.__name__} "{ep_str.decode()}">'

    def remote_endpoint_type(self) -> RemoteEndpointType:
        cdef RemoteEndpointType result
        with nogil:
            result = deref(self._handle).remote_endpoint_type()
        return result

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
