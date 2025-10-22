# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# distutils: language = c++
# cython: language_level=3


from libcpp cimport bool


cdef extern from "<kvikio/shim/cufile.hpp>" nogil:
    cdef int cpp_libcufile_version "kvikio::cufile_version"() except +
    cdef void cpp_driver_open "kvikio::cuFileAPI::instance().driver_open"() except +
    cdef void cpp_driver_close "kvikio::cuFileAPI::instance().driver_close"() except +


def libcufile_version() -> int:
    cdef int version
    with nogil:
        version = cpp_libcufile_version()
    return version


def driver_open():
    with nogil:
        cpp_driver_open()


def driver_close():
    with nogil:
        cpp_driver_close()


cdef extern from "<kvikio/cufile/driver.hpp>" nogil:
    cdef cppclass cpp_DriverProperties "kvikio::DriverProperties":
        cpp_DriverProperties() except +
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


cdef class DriverProperties:
    cdef cpp_DriverProperties _handle

    @property
    def is_gds_available(self) -> bool:
        cdef bool result
        try:
            with nogil:
                result = self._handle.is_gds_available()
            return result
        except RuntimeError:
            return False

    @property
    def major_version(self) -> bool:
        cdef unsigned int version
        with nogil:
            version = self._handle.get_nvfs_major_version()
        return version

    @property
    def minor_version(self) -> bool:
        cdef unsigned int version
        with nogil:
            version = self._handle.get_nvfs_minor_version()
        return version

    @property
    def allow_compat_mode(self) -> bool:
        cdef bool result
        with nogil:
            result = self._handle.get_nvfs_allow_compat_mode()
        return result

    @property
    def poll_mode(self) -> bool:
        cdef bool result
        with nogil:
            result = self._handle.get_nvfs_poll_mode()
        return result

    @poll_mode.setter
    def poll_mode(self, enable: bool) -> None:
        cdef bool cpp_enable = enable
        with nogil:
            self._handle.set_nvfs_poll_mode(cpp_enable)

    @property
    def poll_thresh_size(self) -> int:
        cdef size_t size
        with nogil:
            size = self._handle.get_nvfs_poll_thresh_size()
        return size

    @poll_thresh_size.setter
    def poll_thresh_size(self, size_in_kb: int) -> None:
        cdef size_t size = size_in_kb
        with nogil:
            self._handle.set_nvfs_poll_thresh_size(size)

    @property
    def max_device_cache_size(self) -> int:
        cdef size_t size
        with nogil:
            size = self._handle.get_max_device_cache_size()
        return size

    @max_device_cache_size.setter
    def max_device_cache_size(self, size_in_kb: int) -> None:
        cdef size_t size = size_in_kb
        with nogil:
            self._handle.set_max_device_cache_size(size)

    @property
    def per_buffer_cache_size(self) -> int:
        cdef size_t size
        with nogil:
            size = self._handle.get_per_buffer_cache_size()
        return size

    @property
    def max_pinned_memory_size(self) -> int:
        cdef size_t size
        with nogil:
            size = self._handle.get_max_pinned_memory_size()
        return size

    @max_pinned_memory_size.setter
    def max_pinned_memory_size(self, size_in_kb: int) -> None:
        cdef size_t size = size_in_kb
        with nogil:
            self._handle.set_max_pinned_memory_size(size)
