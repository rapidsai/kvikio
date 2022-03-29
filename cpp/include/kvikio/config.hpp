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
#include <stdexcept>
#include <string>
#include <utility>

#include <kvikio/shim/cufile.hpp>

namespace kvikio::config {
namespace {

template <typename T>
T getenv_or(std::string_view env_var_name, T default_val)
{
  const auto* env_val = std::getenv(env_var_name.data());
  if (env_val == nullptr) { return default_val; }

  std::stringstream sstream(env_val);
  T converted_val;
  sstream >> converted_val;
  return converted_val;
}

template <>
bool getenv_or(std::string_view env_var_name, bool default_val)
{
  const auto* env_val = std::getenv(env_var_name.data());
  if (env_val == nullptr) { return default_val; }
  try {
    // Try parsing `env_var_name` as a integer
    return static_cast<bool>(std::stoi(env_val));
  } catch (std::invalid_argument) {
  }
  // Convert to lowercase
  std::string str{env_val};
  std::transform(str.begin(), str.end(), str.begin(), ::tolower);
  // Trim whitespaces
  std::stringstream trimmer;
  trimmer << str;
  str.clear();
  trimmer >> str;
  // Match value
  if (str == "true" || str == "on" || str == "yes") { return true; }
  if (str == "false" || str == "off" || str == "no") { return false; }
  throw std::invalid_argument("Unknown config value: " + std::string{env_val});
}

inline bool _get_global_compat_mode()
{
  if (std::getenv("KVIKIO_COMPAT_MODE") != nullptr) {
    // Setting `KVIKIO_COMPAT_MODE` take precedence
    return getenv_or("KVIKIO_COMPAT_MODE", false);
  }
  // If `KVIKIO_COMPAT_MODE` isn't set, we infer based on runtime environment
  return !is_cufile_library_available() || is_running_in_wsl() || !run_udev_readable();
}

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
 * By default, compatibility mode is enabled:
 *  - when `libcufile` cannot be found
 *  - when running in Windows Subsystem for Linux (WSL)
 *  - when `/run/udev` isn't readable, which typically happens when running inside a docker
 *    image not launched with `--volume /run/udev:/run/udev:ro`
 *
 * @return The boolean answer
 */
inline bool get_global_compat_mode()
{
  static bool ret = _get_global_compat_mode();
  return ret;
}

}  // namespace kvikio::config
