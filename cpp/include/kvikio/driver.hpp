/*
 * Copyright (c) 2021-2023, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include <iostream>
#include <vector>

#include <kvikio/error.hpp>
#include <kvikio/shim/cufile.hpp>
#include <kvikio/shim/cufile_h_wrapper.hpp>

namespace kvikio {
namespace detail {

[[nodiscard]] inline bool get_driver_flag(unsigned int prop, unsigned int flag) noexcept
{
  return (prop & (1U << flag)) != 0;
}

inline void set_driver_flag(unsigned int& prop, unsigned int flag, bool val) noexcept
{
  if (val) {
    prop |= (1U << flag);
  } else {
    prop &= ~(1U << flag);
  }
}
}  // namespace detail

#ifdef KVIKIO_CUFILE_FOUND

class DriverInitializer {
  // Optional, if not used cuFiles opens the driver automatically
 public:
  DriverInitializer() { CUFILE_TRY(cuFileAPI::instance().DriverOpen()); }

  DriverInitializer(DriverInitializer const&)                = delete;
  DriverInitializer& operator=(DriverInitializer const&)     = delete;
  DriverInitializer(DriverInitializer&&) noexcept            = delete;
  DriverInitializer& operator=(DriverInitializer&&) noexcept = delete;

  ~DriverInitializer()
  {
    try {
      CUFILE_TRY(cuFileAPI::instance().DriverClose());
    } catch (const CUfileException& e) {
      std::cerr << "Unable to close GDS file driver: ";
      std::cerr << e.what();
      std::cerr << std::endl;
    }
  }
};

class DriverProperties {
 private:
  CUfileDrvProps_t _props{};
  bool _initialized{false};

  // Because Cython does not handle exceptions in the default
  // constructor, we initialize `_props` lazily.
  void lazy_init()
  {
    if (_initialized) { return; }
    _initialized = true;
    CUFILE_TRY(cuFileAPI::instance().DriverGetProperties(&_props));
  }

 public:
  DriverProperties() = default;

  bool is_gds_available()
  {
    // If both the major and minor version is zero, the GDS driver isn't loaded.
    return !(get_nvfs_major_version() == 0 && get_nvfs_minor_version() == 0);
  }

  [[nodiscard]] unsigned int get_nvfs_major_version()
  {
    lazy_init();
    return _props.nvfs.major_version;
  }

  [[nodiscard]] unsigned int get_nvfs_minor_version()
  {
    lazy_init();
    return _props.nvfs.minor_version;
  }

  [[nodiscard]] bool get_nvfs_allow_compat_mode()
  {
    lazy_init();
    return detail::get_driver_flag(_props.nvfs.dcontrolflags, CU_FILE_ALLOW_COMPAT_MODE);
  }

  [[nodiscard]] bool get_nvfs_poll_mode()
  {
    lazy_init();
    return detail::get_driver_flag(_props.nvfs.dcontrolflags, CU_FILE_USE_POLL_MODE);
  }

  [[nodiscard]] std::size_t get_nvfs_poll_thresh_size()
  {
    lazy_init();
    return _props.nvfs.poll_thresh_size;
  }

  void set_nvfs_poll_mode(bool enable)
  {
    lazy_init();
    CUFILE_TRY(cuFileAPI::instance().DriverSetPollMode(enable, get_nvfs_poll_thresh_size()));
    detail::set_driver_flag(_props.nvfs.dcontrolflags, CU_FILE_USE_POLL_MODE, enable);
  }

  void set_nvfs_poll_thresh_size(std::size_t size_in_kb)
  {
    lazy_init();
    CUFILE_TRY(cuFileAPI::instance().DriverSetPollMode(get_nvfs_poll_mode(), size_in_kb));
    _props.nvfs.poll_thresh_size = size_in_kb;
  }

  [[nodiscard]] std::vector<CUfileDriverControlFlags> get_nvfs_statusflags()
  {
    lazy_init();
    std::vector<CUfileDriverControlFlags> ret;
    if (detail::get_driver_flag(_props.nvfs.dcontrolflags, CU_FILE_USE_POLL_MODE)) {
      ret.push_back(CU_FILE_USE_POLL_MODE);
    }
    if (detail::get_driver_flag(_props.nvfs.dcontrolflags, CU_FILE_ALLOW_COMPAT_MODE)) {
      ret.push_back(CU_FILE_ALLOW_COMPAT_MODE);
    }
    return ret;
  }

  [[nodiscard]] std::size_t get_max_device_cache_size()
  {
    lazy_init();
    return _props.max_device_cache_size;
  }

  void set_max_device_cache_size(std::size_t size_in_kb)
  {
    lazy_init();
    CUFILE_TRY(cuFileAPI::instance().DriverSetMaxCacheSize(size_in_kb));
    _props.max_device_cache_size = size_in_kb;
  }

  [[nodiscard]] std::size_t get_per_buffer_cache_size()
  {
    lazy_init();
    return _props.per_buffer_cache_size;
  }

  [[nodiscard]] std::size_t get_max_pinned_memory_size()
  {
    lazy_init();
    return _props.max_device_pinned_mem_size;
  }

  void set_max_pinned_memory_size(std::size_t size_in_kb)
  {
    lazy_init();
    CUFILE_TRY(cuFileAPI::instance().DriverSetMaxPinnedMemSize(size_in_kb));
    _props.max_device_pinned_mem_size = size_in_kb;
  }

  [[nodiscard]] std::size_t get_max_batch_io_size()
  {
#ifdef KVIKIO_CUFILE_BATCH_API_FOUND
    lazy_init();
    return _props.max_batch_io_size;
#else
    return 0;
#endif
  }
};

#else
struct DriverInitializer {
  // Implement a non-default constructor to avoid `unused variable` warnings downstream
  DriverInitializer() {}
};

struct DriverProperties {
  // Implement a non-default constructor to avoid `unused variable` warnings downstream
  DriverProperties() {}

  static bool is_gds_available() { return false; }

  [[nodiscard]] static unsigned int get_nvfs_major_version()
  {
    throw CUfileException("KvikIO not compiled with cuFile.h");
  }

  [[nodiscard]] static unsigned int get_nvfs_minor_version()
  {
    throw CUfileException("KvikIO not compiled with cuFile.h");
  }

  [[nodiscard]] static bool get_nvfs_allow_compat_mode()
  {
    throw CUfileException("KvikIO not compiled with cuFile.h");
  }

  [[nodiscard]] static bool get_nvfs_poll_mode()
  {
    throw CUfileException("KvikIO not compiled with cuFile.h");
  }

  [[nodiscard]] static std::size_t get_nvfs_poll_thresh_size()
  {
    throw CUfileException("KvikIO not compiled with cuFile.h");
  }

  static void set_nvfs_poll_mode(bool enable)
  {
    throw CUfileException("KvikIO not compiled with cuFile.h");
  }

  static void set_nvfs_poll_thresh_size(std::size_t size_in_kb)
  {
    throw CUfileException("KvikIO not compiled with cuFile.h");
  }

  [[nodiscard]] static std::vector<CUfileDriverControlFlags> get_nvfs_statusflags()
  {
    throw CUfileException("KvikIO not compiled with cuFile.h");
  }

  [[nodiscard]] static std::size_t get_max_device_cache_size()
  {
    throw CUfileException("KvikIO not compiled with cuFile.h");
  }

  static void set_max_device_cache_size(std::size_t size_in_kb)
  {
    throw CUfileException("KvikIO not compiled with cuFile.h");
  }

  [[nodiscard]] static std::size_t get_per_buffer_cache_size()
  {
    throw CUfileException("KvikIO not compiled with cuFile.h");
  }

  [[nodiscard]] static std::size_t get_max_pinned_memory_size()
  {
    throw CUfileException("KvikIO not compiled with cuFile.h");
  }

  static void set_max_pinned_memory_size(std::size_t size_in_kb)
  {
    throw CUfileException("KvikIO not compiled with cuFile.h");
  }

  [[nodiscard]] std::size_t get_max_batch_io_size()
  {
    throw CUfileException("KvikIO not compiled with cuFile.h");
  }
};
#endif

}  // namespace kvikio
