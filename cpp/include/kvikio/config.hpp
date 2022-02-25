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
  // TODO: probe cuFile to determent if GDS is available.
  return false;
}

}  // namespace kvikio::config
