/*
 * Copyright (c) 2021-2025, NVIDIA CORPORATION.
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

#include <vector>

#include <kvikio/shim/cufile.hpp>
#include <kvikio/shim/cufile_h_wrapper.hpp>

namespace kvikio {

#ifdef KVIKIO_CUFILE_FOUND

class DriverInitializer {
  // Optional, if not used cuFiles opens the driver automatically
 public:
  DriverInitializer();

  DriverInitializer(DriverInitializer const&)                = delete;
  DriverInitializer& operator=(DriverInitializer const&)     = delete;
  DriverInitializer(DriverInitializer&&) noexcept            = delete;
  DriverInitializer& operator=(DriverInitializer&&) noexcept = delete;

  ~DriverInitializer() noexcept;
};

class DriverProperties {
 private:
  CUfileDrvProps_t _props{};
  bool _initialized{false};

  // Because Cython does not handle exceptions in the default
  // constructor, we initialize `_props` lazily.
  void lazy_init();

 public:
  DriverProperties() = default;

  bool is_gds_available();

  [[nodiscard]] unsigned int get_nvfs_major_version();

  [[nodiscard]] unsigned int get_nvfs_minor_version();

  [[nodiscard]] bool get_nvfs_allow_compat_mode();

  [[nodiscard]] bool get_nvfs_poll_mode();

  [[nodiscard]] std::size_t get_nvfs_poll_thresh_size();

  void set_nvfs_poll_mode(bool enable);

  void set_nvfs_poll_thresh_size(std::size_t size_in_kb);

  [[nodiscard]] std::vector<CUfileDriverControlFlags> get_nvfs_statusflags();

  [[nodiscard]] std::size_t get_max_device_cache_size();

  void set_max_device_cache_size(std::size_t size_in_kb);

  [[nodiscard]] std::size_t get_per_buffer_cache_size();

  [[nodiscard]] std::size_t get_max_pinned_memory_size();

  void set_max_pinned_memory_size(std::size_t size_in_kb);

  [[nodiscard]] std::size_t get_max_batch_io_size();
};

#else
struct DriverInitializer {
  // Implement a non-default constructor to avoid `unused variable` warnings downstream
  DriverInitializer();
};

struct DriverProperties {
  // Implement a non-default constructor to avoid `unused variable` warnings downstream
  DriverProperties();

  static bool is_gds_available();

  [[nodiscard]] static unsigned int get_nvfs_major_version();

  [[nodiscard]] static unsigned int get_nvfs_minor_version();

  [[nodiscard]] static bool get_nvfs_allow_compat_mode();

  [[nodiscard]] static bool get_nvfs_poll_mode();

  [[nodiscard]] static std::size_t get_nvfs_poll_thresh_size();

  static void set_nvfs_poll_mode(bool enable);

  static void set_nvfs_poll_thresh_size(std::size_t size_in_kb);

  [[nodiscard]] static std::vector<CUfileDriverControlFlags> get_nvfs_statusflags();

  [[nodiscard]] static std::size_t get_max_device_cache_size();

  static void set_max_device_cache_size(std::size_t size_in_kb);

  [[nodiscard]] static std::size_t get_per_buffer_cache_size();

  [[nodiscard]] static std::size_t get_max_pinned_memory_size();

  static void set_max_pinned_memory_size(std::size_t size_in_kb);

  [[nodiscard]] std::size_t get_max_batch_io_size();
};
#endif

}  // namespace kvikio
