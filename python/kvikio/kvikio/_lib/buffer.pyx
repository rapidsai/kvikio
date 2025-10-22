# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# distutils: language = c++
# cython: language_level=3


from kvikio._lib.arr cimport Array


cdef extern from "<kvikio/buffer.hpp>" nogil:
    void cpp_memory_register "kvikio::memory_register"(const void* devPtr) except +
    void cpp_memory_deregister "kvikio::memory_deregister"(const void* devPtr) except +


def memory_register(buf) -> None:
    if not isinstance(buf, Array):
        buf = Array(buf)
    cdef Array arr = buf
    with nogil:
        cpp_memory_register(<void*>arr.ptr)


def memory_deregister(buf) -> None:
    if not isinstance(buf, Array):
        buf = Array(buf)
    cdef Array arr = buf
    with nogil:
        cpp_memory_deregister(<void*>arr.ptr)


cdef extern from "<kvikio/bounce_buffer.hpp>" nogil:
    size_t cpp_alloc_retain_clear "kvikio::AllocRetain::instance().clear"() except +


def bounce_buffer_free() -> int:
    cdef size_t result
    with nogil:
        result = cpp_alloc_retain_clear()
    return result
