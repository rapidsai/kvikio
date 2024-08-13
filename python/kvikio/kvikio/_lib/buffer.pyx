# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.

# distutils: language = c++
# cython: language_level=3


from kvikio._lib.arr cimport Array


cdef extern from "<kvikio/buffer.hpp>" namespace "kvikio" nogil:
    void cpp_memory_register "kvikio::memory_register"(const void* devPtr) except +
    void cpp_memory_deregister "kvikio::memory_deregister"(const void* devPtr) except +


def memory_register(buf) -> None:
    if not isinstance(buf, Array):
        buf = Array(buf)
    cdef Array arr = buf
    cpp_memory_register(<void*>arr.ptr)


def memory_deregister(buf) -> None:
    if not isinstance(buf, Array):
        buf = Array(buf)
    cdef Array arr = buf
    cpp_memory_deregister(<void*>arr.ptr)
