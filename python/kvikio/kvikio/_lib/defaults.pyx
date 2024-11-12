# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.

# distutils: language = c++
# cython: language_level=3

from libc.stdint cimport uint8_t
from libcpp cimport bool


cdef extern from "<kvikio/defaults.hpp>" namespace "kvikio" nogil:
    cpdef enum class CompatMode(uint8_t):
        OFF = 0
        ON = 1
        AUTO = 2
    CompatMode cpp_compat_mode "kvikio::defaults::compat_mode"() except +
    void cpp_compat_mode_reset \
        "kvikio::defaults::compat_mode_reset"(CompatMode compat_mode) except +
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


def compat_mode() -> CompatMode:
    return cpp_compat_mode()


def compat_mode_reset(compat_mode: CompatMode) -> None:
    cpp_compat_mode_reset(compat_mode)


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
