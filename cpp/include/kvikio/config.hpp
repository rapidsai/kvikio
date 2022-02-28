/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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

#include <algorithm>
#include <cstdlib>
#include <sstream>
#include <string>
#include <utility>

#include <kvikio/shim/cufile.hpp>

namespace kvikio::config {  // TODO: should this be a singletone class instead?
namespace {

inline bool str_to_boolean(std::string str)
{
  try {
    // Try parsing `str` as a integer
    return static_cast<bool>(std::stoi(str));
  } catch (...) {
  }
  // Convert to lowercase
  std::transform(str.begin(), str.end(), str.begin(), ::tolower);
  // Try parsing `str` using std::boolalpha, which support "true" and "false"
  bool ret = false;
  std::istringstream(str) >> std::boolalpha >> ret;
  return ret;
}

inline int _get_compat_mode_from_env()
{
  const char* str = std::getenv("KVIKIO_COMPAT_MODE");
  if (str == nullptr) { return -1; }
  return static_cast<int>(str_to_boolean(str));
}

/*NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)*/
inline std::pair<bool, bool> _current_global_compat_mode{std::make_pair(false, false)};

}  // namespace

/**
 * @brief Return whether the KvikIO library is running in compatibility mode or not
 *
 * Notice, this is not the same as the compatibility mode in cuFile. That is,
 * cuFile can run in compatibility mode while KvikIO is not.
 *
 * When KvikIO is running in compatibility mode, it doesn't load `libcufile.so`. Instead,
 * reads and writes are done using POSIX.
 *
 * Set the enviornment variable `KVIKIO_COMPAT_MODE` to enable/disable compatibility mode.
 * By default, compatibility mode is enabled when `libcufile` cannot be found.
 *
 * @return The boolean answer
 */
inline bool get_global_compat_mode()
{
  auto [initalized, value] = _current_global_compat_mode;
  if (initalized) { return value; }
  int env = _get_compat_mode_from_env();
  if (env != -1) {
    // Setting `KVIKIO_COMPAT_MODE` take precedence
    return static_cast<bool>(env);
  }
  // TODO: check if running in an enviornment not compabtile with cuFile, such as WSL
  //       see <https://github.com/rapidsai/kvikio/issues/11>

  return !is_cufile_library_available();
}

}  // namespace kvikio::config
