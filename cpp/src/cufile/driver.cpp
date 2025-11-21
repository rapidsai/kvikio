/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <iostream>
#include <vector>

#include <kvikio/cufile/driver.hpp>
#include <kvikio/error.hpp>
#include <kvikio/shim/cufile.hpp>
#include <kvikio/shim/cufile_h_wrapper.hpp>

namespace kvikio {
namespace {

[[nodiscard]] bool get_driver_flag(unsigned int prop, unsigned int flag) noexcept
{
  return (prop & (1U << flag)) != 0;
}

void set_driver_flag(unsigned int& prop, unsigned int flag, bool val) noexcept
{
  if (val) {
    prop |= (1U << flag);
  } else {
    prop &= ~(1U << flag);
  }
}
}  // namespace

#ifdef KVIKIO_CUFILE_FOUND

DriverInitializer::DriverInitializer() { cuFileAPI::instance().driver_open(); }

DriverInitializer::~DriverInitializer() noexcept
{
  try {
    cuFileAPI::instance().driver_close();
  } catch (CUfileException const& e) {
    std::cerr << "Unable to close GDS file driver: ";
    std::cerr << e.what();
    std::cerr << std::endl;
  }
}

// Because Cython does not handle exceptions in the default
// constructor, we initialize `_props` lazily.
void DriverProperties::lazy_init()
{
  if (_initialized) { return; }
  _initialized = true;
  CUFILE_TRY(cuFileAPI::instance().DriverGetProperties(&_props));
}

bool DriverProperties::is_gds_available()
{
  // If both the major and minor version is zero, the GDS driver isn't loaded.
  return !(get_nvfs_major_version() == 0 && get_nvfs_minor_version() == 0);
}

unsigned int DriverProperties::get_nvfs_major_version()
{
  lazy_init();
  return _props.nvfs.major_version;
}

unsigned int DriverProperties::get_nvfs_minor_version()
{
  lazy_init();
  return _props.nvfs.minor_version;
}

bool DriverProperties::get_nvfs_allow_compat_mode()
{
  lazy_init();
  return get_driver_flag(_props.nvfs.dcontrolflags, CU_FILE_ALLOW_COMPAT_MODE);
}

bool DriverProperties::get_nvfs_poll_mode()
{
  lazy_init();
  return get_driver_flag(_props.nvfs.dcontrolflags, CU_FILE_USE_POLL_MODE);
}

std::size_t DriverProperties::get_nvfs_poll_thresh_size()
{
  lazy_init();
  return _props.nvfs.poll_thresh_size;
}

void DriverProperties::set_nvfs_poll_mode(bool enable)
{
  lazy_init();
  CUFILE_TRY(cuFileAPI::instance().DriverSetPollMode(enable, get_nvfs_poll_thresh_size()));
  set_driver_flag(_props.nvfs.dcontrolflags, CU_FILE_USE_POLL_MODE, enable);
}

void DriverProperties::set_nvfs_poll_thresh_size(std::size_t size_in_kb)
{
  lazy_init();
  CUFILE_TRY(cuFileAPI::instance().DriverSetPollMode(get_nvfs_poll_mode(), size_in_kb));
  _props.nvfs.poll_thresh_size = size_in_kb;
}

std::vector<CUfileDriverControlFlags> DriverProperties::get_nvfs_statusflags()
{
  lazy_init();
  std::vector<CUfileDriverControlFlags> ret;
  if (get_driver_flag(_props.nvfs.dcontrolflags, CU_FILE_USE_POLL_MODE)) {
    ret.push_back(CU_FILE_USE_POLL_MODE);
  }
  if (get_driver_flag(_props.nvfs.dcontrolflags, CU_FILE_ALLOW_COMPAT_MODE)) {
    ret.push_back(CU_FILE_ALLOW_COMPAT_MODE);
  }
  return ret;
}

std::size_t DriverProperties::get_max_device_cache_size()
{
  lazy_init();
  return _props.max_device_cache_size;
}

void DriverProperties::set_max_device_cache_size(std::size_t size_in_kb)
{
  lazy_init();
  CUFILE_TRY(cuFileAPI::instance().DriverSetMaxCacheSize(size_in_kb));
  _props.max_device_cache_size = size_in_kb;
}

std::size_t DriverProperties::get_per_buffer_cache_size()
{
  lazy_init();
  return _props.per_buffer_cache_size;
}

std::size_t DriverProperties::get_max_pinned_memory_size()
{
  lazy_init();
  return _props.max_device_pinned_mem_size;
}

void DriverProperties::set_max_pinned_memory_size(std::size_t size_in_kb)
{
  lazy_init();
  CUFILE_TRY(cuFileAPI::instance().DriverSetMaxPinnedMemSize(size_in_kb));
  _props.max_device_pinned_mem_size = size_in_kb;
}

std::size_t DriverProperties::get_max_batch_io_size()
{
#ifdef KVIKIO_CUFILE_BATCH_API_FOUND
  lazy_init();
  return _props.max_batch_io_size;
#else
  return 0;
#endif
}

#else
DriverInitializer::DriverInitializer() {}

DriverProperties::DriverProperties() {}

bool DriverProperties::is_gds_available() { return false; }

unsigned int DriverProperties::get_nvfs_major_version()
{
  KVIKIO_FAIL("KvikIO not compiled with cuFile.h");
  return {};
}

unsigned int DriverProperties::get_nvfs_minor_version()
{
  KVIKIO_FAIL("KvikIO not compiled with cuFile.h");
  return {};
}

bool DriverProperties::get_nvfs_allow_compat_mode()
{
  KVIKIO_FAIL("KvikIO not compiled with cuFile.h");
  return {};
}

bool DriverProperties::get_nvfs_poll_mode()
{
  KVIKIO_FAIL("KvikIO not compiled with cuFile.h");
  return {};
}

std::size_t DriverProperties::get_nvfs_poll_thresh_size()
{
  KVIKIO_FAIL("KvikIO not compiled with cuFile.h");
  return {};
}

void DriverProperties::set_nvfs_poll_mode(bool enable)
{
  KVIKIO_FAIL("KvikIO not compiled with cuFile.h");
  return;
}

void DriverProperties::set_nvfs_poll_thresh_size(std::size_t size_in_kb)
{
  KVIKIO_FAIL("KvikIO not compiled with cuFile.h");
  return;
}

std::vector<CUfileDriverControlFlags> DriverProperties::get_nvfs_statusflags()
{
  KVIKIO_FAIL("KvikIO not compiled with cuFile.h");
  return {};
}

std::size_t DriverProperties::get_max_device_cache_size()
{
  KVIKIO_FAIL("KvikIO not compiled with cuFile.h");
  return {};
}

void DriverProperties::set_max_device_cache_size(std::size_t size_in_kb)
{
  KVIKIO_FAIL("KvikIO not compiled with cuFile.h");
  return;
}

std::size_t DriverProperties::get_per_buffer_cache_size()
{
  KVIKIO_FAIL("KvikIO not compiled with cuFile.h");
  return {};
}

std::size_t DriverProperties::get_max_pinned_memory_size()
{
  KVIKIO_FAIL("KvikIO not compiled with cuFile.h");
  return {};
}

void DriverProperties::set_max_pinned_memory_size(std::size_t size_in_kb)
{
  KVIKIO_FAIL("KvikIO not compiled with cuFile.h");
  return;
}

std::size_t DriverProperties::get_max_batch_io_size()
{
  KVIKIO_FAIL("KvikIO not compiled with cuFile.h");
  return {};
}
#endif

}  // namespace kvikio
