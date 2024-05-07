# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.

# distutils: language = c++
# cython: language_level=3

from posix cimport fcntl

from libcpp cimport bool
from libcpp.string cimport string
from libcpp.utility cimport pair
from libcpp.vector cimport vector


cdef extern from "cuda.h":
    ctypedef void* CUstream


cdef extern from "<future>" namespace "std" nogil:
    cdef cppclass future[T]:
        future() except +
        T get() except +


cdef extern from "<kvikio/stream.hpp>" namespace "kvikio" nogil:
    cdef cppclass StreamFuture:
        StreamFuture() except +
        StreamFuture(StreamFuture&&) except +
        size_t check_bytes_done() except +


cdef extern from "<kvikio/utils.hpp>" namespace "kvikio" nogil:
    bool is_future_done[T](const T& future) except +


cdef extern from "<kvikio/driver.hpp>" namespace "kvikio" nogil:
    cdef cppclass DriverProperties:
        DriverProperties() except +
        bool is_gds_available() except +
        unsigned int get_nvfs_major_version() except +
        unsigned int get_nvfs_minor_version() except +
        bool get_nvfs_allow_compat_mode() except +
        bool get_nvfs_poll_mode() except +
        size_t get_nvfs_poll_thresh_size() except +
        void set_nvfs_poll_mode(bool enable) except +
        void set_nvfs_poll_thresh_size(size_t size_in_kb) except +
        size_t get_max_device_cache_size() except +
        void set_max_device_cache_size(size_t size_in_kb) except +
        size_t get_per_buffer_cache_size() except +
        size_t get_max_pinned_memory_size() except +
        void set_max_pinned_memory_size(size_t size_in_kb) except +


cdef extern from "<kvikio/buffer.hpp>" namespace "kvikio" nogil:
    void memory_register(const void* devPtr) except +
    void memory_deregister(const void* devPtr) except +


cdef extern from "<kvikio/defaults.hpp>" namespace "kvikio::defaults" nogil:
    bool compat_mode() except +
    void compat_mode_reset(bool enable) except +
    unsigned int thread_pool_nthreads() except +
    void thread_pool_nthreads_reset(unsigned int nthreads) except +
    size_t task_size() except +
    void task_size_reset(size_t nbytes) except +
    size_t gds_threshold() except +
    void gds_threshold_reset(size_t nbytes) except +


cdef extern from "<kvikio/file_handle.hpp>" namespace "kvikio" nogil:
    cdef cppclass FileHandle:
        FileHandle() except +
        FileHandle(int fd) except +
        FileHandle(
            string file_path,
            string flags,
        ) except +
        FileHandle(
            string file_path,
            string flags,
            fcntl.mode_t mode
        ) except +
        void close()
        bool closed()
        int fd()
        int fd_open_flags() except +
        future[size_t] pread(
            void* devPtr,
            size_t size,
            size_t file_offset,
            size_t task_size
        ) except +
        future[size_t] pwrite(
            void* devPtr,
            size_t size,
            size_t file_offset,
            size_t task_size
        ) except +
        size_t read(
            void* devPtr_base,
            size_t size,
            size_t file_offset,
            size_t devPtr_offset
        ) except +
        size_t write(
            void* devPtr_base,
            size_t size,
            size_t file_offset,
            size_t devPtr_offset
        ) except +
        StreamFuture read_async(
            void* devPtr_base,
            size_t size,
            size_t file_offset,
            size_t devPtr_offset,
            CUstream stream
        ) except +
        StreamFuture write_async(
            void* devPtr_base,
            size_t size,
            size_t file_offset,
            size_t devPtr_offset,
            CUstream stream
        ) except +
