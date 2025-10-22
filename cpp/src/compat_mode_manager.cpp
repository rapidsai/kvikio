/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cassert>
#include <stdexcept>

#include <kvikio/compat_mode_manager.hpp>
#include <kvikio/cufile/config.hpp>
#include <kvikio/defaults.hpp>
#include <kvikio/detail/nvtx.hpp>
#include <kvikio/error.hpp>
#include <kvikio/file_handle.hpp>
#include <kvikio/shim/cufile.hpp>

namespace kvikio {

bool CompatModeManager::is_compat_mode_preferred() const noexcept
{
  return _is_compat_mode_preferred;
}

bool CompatModeManager::is_compat_mode_preferred_for_async() const noexcept
{
  return _is_compat_mode_preferred_for_async;
}

CompatMode CompatModeManager::compat_mode_requested() const noexcept
{
  return _compat_mode_requested;
}

CompatModeManager::CompatModeManager(std::string const& file_path,
                                     std::string const& flags,
                                     mode_t mode,
                                     CompatMode compat_mode_requested_v,
                                     FileHandle* file_handle)
{
  KVIKIO_NVTX_FUNC_RANGE();
  KVIKIO_EXPECT(file_handle != nullptr,
                "The compatibility mode manager does not have a proper owning file handle.",
                std::invalid_argument);

  _compat_mode_requested = compat_mode_requested_v;
  file_handle->_file_direct_off.open(file_path, flags, false, mode);
  _is_compat_mode_preferred = defaults::is_compat_mode_preferred(compat_mode_requested_v);

  // Nothing to do in compatibility mode
  if (_is_compat_mode_preferred) { return; }

  try {
    file_handle->_file_direct_on.open(file_path, flags, true, mode);
  } catch (...) {
    // Try to open the file with the O_DIRECT flag. Fall back to compatibility mode, if it fails.
    if (compat_mode_requested_v == CompatMode::AUTO) {
      _is_compat_mode_preferred = true;
    } else {  // CompatMode::OFF
      throw;
    }
  }

  if (_is_compat_mode_preferred) { return; }

  auto error_code = file_handle->_cufile_handle.register_handle(file_handle->_file_direct_on.fd());
  assert(error_code.has_value());

  // For the AUTO mode, if the first cuFile API call fails, fall back to the compatibility
  // mode.
  if (compat_mode_requested_v == CompatMode::AUTO && error_code.value().err != CU_FILE_SUCCESS) {
    _is_compat_mode_preferred = true;
  } else {
    CUFILE_TRY(error_code.value());
  }

  // Check cuFile async API
  static bool const is_extra_symbol_available = is_stream_api_available();
  static bool const is_config_path_empty      = config_path().empty();
  _is_compat_mode_preferred_for_async =
    _is_compat_mode_preferred || !is_extra_symbol_available || is_config_path_empty;
}

void CompatModeManager::validate_compat_mode_for_async() const
{
  KVIKIO_NVTX_FUNC_RANGE();
  if (_is_compat_mode_preferred_for_async && _compat_mode_requested == CompatMode::OFF) {
    std::string err_msg;
    if (!is_stream_api_available()) { err_msg += "Missing the cuFile stream api."; }

    // When checking for availability, we also check if cuFile's config file exists. This is
    // because even when the stream API is available, it doesn't work if no config file exists.
    if (config_path().empty()) { err_msg += " Missing cuFile configuration file."; }

    KVIKIO_FAIL(err_msg, std::runtime_error);
  }
}

}  // namespace kvikio
