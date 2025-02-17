/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
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

#include <algorithm>
#include <cassert>
#include <stdexcept>

#include <kvikio/compat_mode.hpp>
#include <kvikio/cufile/config.hpp>
#include <kvikio/error.hpp>
#include <kvikio/file_handle.hpp>
#include <kvikio/shim/cufile.hpp>

namespace kvikio {

namespace detail {
CompatMode parse_compat_mode_str(std::string_view compat_mode_str)
{
  // Convert to lowercase
  std::string tmp{compat_mode_str};
  std::transform(
    tmp.begin(), tmp.end(), tmp.begin(), [](unsigned char c) { return std::tolower(c); });

  if (tmp == "on" || tmp == "true" || tmp == "yes" || tmp == "1") {
    return CompatMode::ON;
  } else if (tmp == "off" || tmp == "false" || tmp == "no" || tmp == "0") {
    return CompatMode::OFF;
  } else if (tmp == "auto") {
    return CompatMode::AUTO;
  } else {
    throw std::invalid_argument("Unknown compatibility mode: " + std::string{tmp});
  }
}

}  // namespace detail

CompatMode CompatModeManager::infer_compat_mode_if_auto(CompatMode compat_mode) noexcept
{
  if (compat_mode == CompatMode::AUTO) {
    return is_cufile_available() ? CompatMode::OFF : CompatMode::ON;
  }
  return compat_mode;
}

bool CompatModeManager::is_compat_mode_preferred(CompatMode compat_mode) noexcept
{
  return compat_mode == CompatMode::ON ||
         (compat_mode == CompatMode::AUTO &&
          infer_compat_mode_if_auto(compat_mode) == CompatMode::ON);
}

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
  if (file_handle == nullptr) {
    throw std::invalid_argument(
      "The compatibility mode manager does not have a proper owning file handle.");
  }

  file_handle->_file_direct_off.open(file_path, flags, false, mode);
  _is_compat_mode_preferred = is_compat_mode_preferred(compat_mode_requested_v);

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
  static bool is_extra_symbol_available = is_stream_api_available();
  static bool is_config_path_empty      = config_path().empty();
  _is_compat_mode_preferred_for_async =
    _is_compat_mode_preferred || !is_extra_symbol_available || is_config_path_empty;
  return;
}

void CompatModeManager::validate_compat_mode_for_async() const
{
  if (!_is_compat_mode_preferred && _is_compat_mode_preferred_for_async &&
      _compat_mode_requested == CompatMode::OFF) {
    std::string err_msg;
    if (!is_stream_api_available()) { err_msg += "Missing the cuFile stream api."; }

    // When checking for availability, we also check if cuFile's config file exists. This is
    // because even when the stream API is available, it doesn't work if no config file exists.
    if (config_path().empty()) { err_msg += " Missing cuFile configuration file."; }

    throw std::runtime_error(err_msg);
  }
}

}  // namespace kvikio
