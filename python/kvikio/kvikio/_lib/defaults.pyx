# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# distutils: language = c++
# cython: language_level=3

from libc.stdint cimport uint8_t
from libcpp cimport bool
from libcpp.vector cimport vector


cdef extern from "<kvikio/defaults.hpp>" namespace "kvikio" nogil:
    cpdef enum class CompatMode(uint8_t):
        OFF = 0
        ON = 1
        AUTO = 2
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
