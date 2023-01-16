/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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
#include <cstddef>
#include <cstdlib>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

namespace kvikio::detail {

/**
 * @brief Trim and convert string to lowercase
 *
 * @param str The original string
 *
 * @return A copy of `str` that is trimmed and converted to lowercase
 */
std::string trim_and_lowercase(std::string_view str)
{
  // Convert to lowercase
  std::string ret{str};
  std::transform(ret.begin(), ret.end(), ret.begin(), ::tolower);
  // Trim whitespaces
  std::stringstream trimmer;
  trimmer << ret;
  ret.clear();
  trimmer >> ret;
  return ret;
}

/**
 * @brief Check if environment variable is unset or "auto"
 *
 * @param env_var_name The name of the environment variable
 *
 * @return Returns true if unset or set to "auto" otherwise returns false.
 */
inline bool env_unset_or_auto(std::string_view env_var_name)
{
  const auto* env_val = std::getenv(env_var_name.data());
  return env_val == nullptr || trim_and_lowercase(env_val) == "auto";
}

/**
 * @brief Get environment variable or a default value
 *
 * If it doesn't exist, returns `default_val`.
 * If it exist, reads `env_var_name` and interpret the value as of type `T`.
 *
 * @param env_var_name The name of the environment variable
 * @param default_val The default value
 *
 * @return The environment variable or `default_val`
 */
template <typename T>
T getenv_or(std::string_view env_var_name, T default_val)
{
  const auto* env_val = std::getenv(env_var_name.data());
  if (env_val == nullptr) { return default_val; }

  std::stringstream sstream(env_val);
  T converted_val;
  sstream >> converted_val;
  if (sstream.fail()) {
    throw std::invalid_argument("unknown config value " + std::string{env_var_name} + "=" +
                                std::string{env_val});
  }
  return converted_val;
}

/**
 * @brief Get environment variable or a default value
 *
 * If it doesn't exist, returns `default_val`.
 * If it exist, reads `env_var_name` and interpret the value as a boolean:
 *  - True values: ["1", "true", "on", "yes"]
 *  - False values: ["0", "false", "off", "no"]
 * Ignores character cases.
 *
 * @param env_var_name The name of the environment variable
 * @param default_val The default value
 *
 * @return The environment variable or `default_val`
 */
template <>
inline bool getenv_or(std::string_view env_var_name, bool default_val)
{
  const auto* env_val = std::getenv(env_var_name.data());
  if (env_val == nullptr) { return default_val; }
  try {
    // Try parsing `env_var_name` as a integer
    return static_cast<bool>(std::stoi(env_val));
  } catch (const std::invalid_argument&) {
  }

  std::string str = trim_and_lowercase(env_val);
  if (str == "true" || str == "on" || str == "yes") { return true; }
  if (str == "false" || str == "off" || str == "no") { return false; }
  throw std::invalid_argument("unknown config value " + std::string{env_var_name} + "=" +
                              std::string{env_val});
}

}  // namespace kvikio::detail
