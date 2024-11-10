# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.

# distutils: language = c++
# cython: language_level=3

from libcpp cimport bool
from libcpp.string cimport string


cdef extern from "<kvikio/defaults.hpp>" nogil:
    bool cpp_compat_mode "kvikio::defaults::compat_mode"() except +
    void cpp_compat_mode_reset_bool \
        "kvikio::defaults::compat_mode_reset"(bool enable) except +
    void cpp_compat_mode_reset_str \
        "kvikio::defaults::compat_mode_reset"(string compat_mode_str) except +
    unsigned int cpp_thread_pool_nthreads \
        "kvikio::defaults::thread_pool_nthreads"() except +
    void cpp_thread_pool_nthreads_reset \
        "kvikio::defaults::thread_pool_nthreads_reset" (unsigned int nthreads) except +
    size_t cpp_task_size "kvikio::defaults::task_size"() except +
    void cpp_task_size_reset "kvikio::defaults::task_size_reset"(size_t nbytes) except +
    size_t cpp_gds_threshold "kvikio::defaults::gds_threshold"() except +
    void cpp_gds_threshold_reset \
        "kvikio::defaults::gds_threshold_reset"(size_t nbytes) except +
    size_t cpp_bounce_buffer_size "kvikio::defaults::bounce_buffer_size"() except +
    void cpp_bounce_buffer_size_reset \
        "kvikio::defaults::bounce_buffer_size_reset"(size_t nbytes) except +


def compat_mode() -> bool:
    return cpp_compat_mode()


cdef string _to_string(str s):
    """Convert Python object to a C++ string (if None, return the empty string)"""
    if s is not None:
        return s.encode()
    else:
        return string()


def compat_mode_reset_bool(enable: bool) -> None:
    cpp_compat_mode_reset_bool(enable)


def compat_mode_reset_str(compat_mode_str: str) -> None:
    cpp_compat_mode_reset_str(_to_string(compat_mode_str))


def thread_pool_nthreads() -> int:
    return cpp_thread_pool_nthreads()


def thread_pool_nthreads_reset(nthreads: int) -> None:
    cpp_thread_pool_nthreads_reset(nthreads)


def task_size() -> int:
    return cpp_task_size()


def task_size_reset(nbytes: int) -> None:
    cpp_task_size_reset(nbytes)


def gds_threshold() -> int:
    return cpp_gds_threshold()


def gds_threshold_reset(nbytes: int) -> None:
    cpp_gds_threshold_reset(nbytes)


def bounce_buffer_size() -> int:
    return cpp_bounce_buffer_size()


def bounce_buffer_size_reset(nbytes: int) -> None:
    cpp_bounce_buffer_size_reset(nbytes)
