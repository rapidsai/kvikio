# Copyright (c) 2024-2025, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.

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
    bool cpp_split_gds_io "kvikio::defaults::split_gds_io"() except +
    void cpp_set_split_gds_io \
        "kvikio::defaults::set_split_gds_io"(bool flag) except +
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
    return cpp_is_compat_mode_preferred()


def compat_mode() -> CompatMode:
    return cpp_compat_mode()


def set_compat_mode(compat_mode: CompatMode) -> None:
    cpp_set_compat_mode(compat_mode)


def thread_pool_nthreads() -> int:
    return cpp_thread_pool_nthreads()


def set_thread_pool_nthreads(nthreads: int) -> None:
    cpp_set_thread_pool_nthreads(nthreads)


def task_size() -> int:
    return cpp_task_size()


def set_task_size(nbytes: int) -> None:
    cpp_set_task_size(nbytes)


def gds_threshold() -> int:
    return cpp_gds_threshold()


def set_gds_threshold(nbytes: int) -> None:
    cpp_set_gds_threshold(nbytes)


def split_gds_io() -> bool:
    return cpp_split_gds_io()


def set_split_gds_io(flag: bool) -> None:
    cpp_set_split_gds_io(flag)


def bounce_buffer_size() -> int:
    return cpp_bounce_buffer_size()


def set_bounce_buffer_size(nbytes: int) -> None:
    cpp_set_bounce_buffer_size(nbytes)


def http_max_attempts() -> int:
    return cpp_http_max_attempts()


def set_http_max_attempts(attempts: int) -> None:
    cpp_set_http_max_attempts(attempts)


def http_timeout() -> int:
    return cpp_http_timeout()


def set_http_timeout(timeout: int) -> None:
    return cpp_set_http_timeout(timeout)


def http_status_codes() -> list[int]:
    return cpp_http_status_codes()


def set_http_status_codes(status_codes: list[int]) -> None:
    return cpp_set_http_status_codes(status_codes)
