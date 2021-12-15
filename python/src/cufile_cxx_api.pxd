# Copyright (c) 2021-2022, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.

# cython: language_level=3

from posix cimport fcntl

from libcpp cimport bool
from libcpp.string cimport string
from libcpp.utility cimport pair


cdef extern from "cufile/driver.hpp" namespace "cufile" nogil:
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


cdef extern from "cufile/buffer.hpp" namespace "cufile" nogil:
    void memory_register(const void* devPtr) except +
    void memory_deregister(const void* devPtr) except +


cdef extern from "cufile/file_handle.hpp" namespace "cufile::default_thread_pool" nogil:
    void reset(unsigned int nthreads)
    unsigned int nthreads()


cdef extern from "cufile/file_handle.hpp" namespace "cufile" nogil:
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
        size_t pread(
            void *devPtr,
            size_t size,
            size_t file_offset
        ) except +
        size_t pwrite(
            void *devPtr,
            size_t size,
            size_t file_offset
        ) except +


cdef extern from "cufile/nvml.hpp" namespace "cufile" nogil:
    cdef cppclass NVML:
        NVML() except +
        string get_name() except +
        pair[size_t, size_t] get_memory() except +
        pair[size_t, size_t] get_bar1_memory() except +
