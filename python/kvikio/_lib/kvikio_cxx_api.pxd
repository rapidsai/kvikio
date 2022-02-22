# Copyright (c) 2021-2022, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.

# distutils: language = c++
# cython: language_level=3

from posix cimport fcntl

from libcpp cimport bool
from libcpp.string cimport string
from libcpp.utility cimport pair
from libcpp.vector cimport vector


cdef extern from "<future>" namespace "std" nogil:
    cdef cppclass future[T]:
        future() except +
        T get() except +


cdef extern from "<kvikio/utils.hpp>" namespace "kvikio" nogil:
    bool is_future_done[T](const T& future) except +


cdef extern from "<kvikio/driver.hpp>" namespace "kvikio" nogil:
    cdef cppclass DriverProperties:
        DriverProperties() except +
        bool is_gds_availabe() except +
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


cdef extern from "<kvikio/thread_pool/default.hpp>" namespace "kvikio::default_thread_pool" nogil:
    void reset(unsigned int nthreads)
    unsigned int nthreads()


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
            size_t ntasks
        ) except +
        future[size_t] pwrite(
            void* devPtr,
            size_t size,
            size_t file_offset,
            size_t ntasks
        ) except +


cdef extern from "<kvikio/nvml.hpp>" namespace "kvikio" nogil:
    cdef cppclass NVML:
        NVML() except +
        string get_name() except +
        pair[size_t, size_t] get_memory() except +
        pair[size_t, size_t] get_bar1_memory() except +
