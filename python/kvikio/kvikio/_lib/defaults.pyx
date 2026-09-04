# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# distutils: language = c++
# cython: language_level=3

from libc.stdint cimport uint8_t
from libcpp cimport bool
from libcpp.vector cimport vector


cdef extern from "<kvikio/defaults.hpp>" namespace "kvikio" nogil:
    cpdef enum class CompatMode(uint8_t):
        OFF
        ON
        AUTO
    bool cpp_is_compat_mode_preferred \
        "kvikio::defaults::is_compat_mode_preferred"() except +
    CompatMode cpp_compat_mode "kvikio::defaults::compat_mode"() except +
    void cpp_set_compat_mode \
        "kvikio::defaults::set_compat_mode"(CompatMode compat_mode) except +
    unsigned int cpp_thread_pool_nthreads \
        "kvikio::defaults::thread_pool_nthreads"() except +
    void cpp_set_thread_pool_nthreads \
        "kvikio::defaults::set_thread_pool_nthreads" (unsigned int nthreads) except +
    size_t cpp_task_size "kvikio::defaults::task_size"() except +
    void cpp_set_task_size "kvikio::defaults::set_task_size"(size_t nbytes) except +
    size_t cpp_gds_threshold "kvikio::defaults::gds_threshold"() except +
    void cpp_set_gds_threshold \
        "kvikio::defaults::set_gds_threshold"(size_t nbytes) except +
    size_t cpp_bounce_buffer_size "kvikio::defaults::bounce_buffer_size"() except +
    void cpp_set_bounce_buffer_size \
        "kvikio::defaults::set_bounce_buffer_size"(size_t nbytes) except +
    size_t cpp_http_max_attempts "kvikio::defaults::http_max_attempts"() except +
    void cpp_set_http_max_attempts \
        "kvikio::defaults::set_http_max_attempts"(size_t attempts) except +
    vector[int] cpp_http_status_codes "kvikio::defaults::http_status_codes"() except +
    void cpp_set_http_status_codes \
        "kvikio::defaults::set_http_status_codes"(vector[int] status_codes) except +
    long cpp_http_timeout "kvikio::defaults::http_timeout"() except +
    void cpp_set_http_timeout\
        "kvikio::defaults::set_http_timeout"(long timeout_seconds) except +
    bool cpp_auto_direct_io_read "kvikio::defaults::auto_direct_io_read"() except +
    void cpp_set_auto_direct_io_read \
        "kvikio::defaults::set_auto_direct_io_read"(size_t flag) except +
    bool cpp_auto_direct_io_write "kvikio::defaults::auto_direct_io_write"() except +
    void cpp_set_auto_direct_io_write \
        "kvikio::defaults::set_auto_direct_io_write"(size_t flag) except +
    RemoteIOBackend cpp_remote_io_backend \
        "kvikio::defaults::remote_io_backend"() except +
    void cpp_set_remote_io_backend \
        "kvikio::defaults::set_remote_io_backend"(RemoteIOBackend backend) except +
    unsigned int cpp_remote_io_num_reactors \
        "kvikio::defaults::remote_io_num_reactors"() except +
    void cpp_set_remote_io_num_reactors \
        "kvikio::defaults::set_remote_io_num_reactors"(unsigned int num_reactors) except +
    RemoteReactorDispatch cpp_remote_io_reactor_dispatch \
        "kvikio::defaults::remote_io_reactor_dispatch"() except +
    void cpp_set_remote_io_reactor_dispatch \
        "kvikio::defaults::set_remote_io_reactor_dispatch"(RemoteReactorDispatch dispatch) except +
    size_t cpp_remote_io_max_concurrent_requests \
        "kvikio::defaults::remote_io_max_concurrent_requests"() except +
    void cpp_set_remote_io_max_concurrent_requests \
        "kvikio::defaults::set_remote_io_max_concurrent_requests"(size_t max_requests) except +


cdef extern from "<kvikio/remote_handle.hpp>" namespace "kvikio" nogil:
    cpdef enum class RemoteIOBackend(uint8_t):
        EASY_THREADPOOL
        MULTI_POLL
    cpdef enum class RemoteReactorDispatch(uint8_t):
        PER_CHUNK
        PER_PREAD


def is_compat_mode_preferred() -> bool:
    cdef bool result
    with nogil:
        result = cpp_is_compat_mode_preferred()
    return result


def compat_mode() -> CompatMode:
    cdef CompatMode result
    with nogil:
        result = cpp_compat_mode()
    return result


def set_compat_mode(compat_mode: CompatMode) -> None:
    with nogil:
        cpp_set_compat_mode(compat_mode)


def thread_pool_nthreads() -> int:
    cdef unsigned int result
    with nogil:
        result = cpp_thread_pool_nthreads()
    return result


def set_thread_pool_nthreads(nthreads: int) -> None:
    cdef unsigned int cpp_nthreads = nthreads
    with nogil:
        cpp_set_thread_pool_nthreads(cpp_nthreads)


def task_size() -> int:
    cdef size_t result
    with nogil:
        result = cpp_task_size()
    return result


def set_task_size(nbytes: int) -> None:
    cdef size_t cpp_nbytes = nbytes
    with nogil:
        cpp_set_task_size(cpp_nbytes)


def gds_threshold() -> int:
    cdef size_t result
    with nogil:
        result = cpp_gds_threshold()
    return result


def set_gds_threshold(nbytes: int) -> None:
    cdef size_t cpp_nbytes = nbytes
    with nogil:
        cpp_set_gds_threshold(cpp_nbytes)


def bounce_buffer_size() -> int:
    cdef size_t result
    with nogil:
        result = cpp_bounce_buffer_size()
    return result


def set_bounce_buffer_size(nbytes: int) -> None:
    cdef size_t cpp_nbytes = nbytes
    with nogil:
        cpp_set_bounce_buffer_size(cpp_nbytes)


def http_max_attempts() -> int:
    cdef size_t result
    with nogil:
        result = cpp_http_max_attempts()
    return result


def set_http_max_attempts(attempts: int) -> None:
    cdef size_t cpp_attempts = attempts
    with nogil:
        cpp_set_http_max_attempts(cpp_attempts)


def http_timeout() -> int:
    cdef long result
    with nogil:
        result = cpp_http_timeout()
    return result


def set_http_timeout(timeout: int) -> None:
    cdef long cpp_timeout = timeout
    with nogil:
        cpp_set_http_timeout(cpp_timeout)


def http_status_codes() -> list[int]:
    # Cannot use nogil here because we need the GIL for list creation
    return cpp_http_status_codes()


def set_http_status_codes(status_codes: list[int]) -> None:
    # Cannot use nogil here because we need the GIL for list conversion
    cpp_set_http_status_codes(status_codes)


def auto_direct_io_read() -> bool:
    cdef bool result
    with nogil:
        result = cpp_auto_direct_io_read()
    return result


def set_auto_direct_io_read(flag: bool) -> None:
    cdef bool cpp_flag = flag
    with nogil:
        cpp_set_auto_direct_io_read(cpp_flag)


def auto_direct_io_write() -> bool:
    cdef bool result
    with nogil:
        result = cpp_auto_direct_io_write()
    return result


def set_auto_direct_io_write(flag: bool) -> None:
    cdef bool cpp_flag = flag
    with nogil:
        cpp_set_auto_direct_io_write(cpp_flag)


def remote_io_backend() -> RemoteIOBackend:
    cdef RemoteIOBackend result
    with nogil:
        result = cpp_remote_io_backend()
    return result


def set_remote_io_backend(backend: RemoteIOBackend) -> None:
    with nogil:
        cpp_set_remote_io_backend(backend)


def remote_io_num_reactors() -> int:
    cdef unsigned int result
    with nogil:
        result = cpp_remote_io_num_reactors()
    return result


def set_remote_io_num_reactors(num_reactors: int) -> None:
    cdef unsigned int cpp_num_reactors = num_reactors
    with nogil:
        cpp_set_remote_io_num_reactors(cpp_num_reactors)


def remote_io_reactor_dispatch() -> RemoteReactorDispatch:
    cdef RemoteReactorDispatch result
    with nogil:
        result = cpp_remote_io_reactor_dispatch()
    return result


def set_remote_io_reactor_dispatch(dispatch: RemoteReactorDispatch) -> None:
    with nogil:
        cpp_set_remote_io_reactor_dispatch(dispatch)


def remote_io_max_concurrent_requests() -> int:
    cdef size_t result
    with nogil:
        result = cpp_remote_io_max_concurrent_requests()
    return result


def set_remote_io_max_concurrent_requests(max_requests: int) -> None:
    cdef size_t cpp_max_requests = max_requests
    with nogil:
        cpp_set_remote_io_max_concurrent_requests(cpp_max_requests)
