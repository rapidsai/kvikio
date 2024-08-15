/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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

#include <cstdlib>
#include <filesystem>
#include <string>

namespace kvikio {
namespace detail {

[[nodiscard]] inline const char* lookup_config_path()
{
  const char* env = std::getenv("CUFILE_ENV_PATH_JSON");
  if (env != nullptr && std::filesystem::exists(env)) { return env; }
  if (std::filesystem::exists("/etc/cufile.json")) { return "/etc/cufile.json"; }
  return "";
}

}  // namespace detail

/**
 * @brief Get the filepath to cuFile's config file (`cufile.json`) or the empty string
 *
 * This lookup is cached.
 *
 * @return The filepath to the cufile.json file or the empty string if it isn't found.
 */
[[nodiscard]] inline const std::string& config_path()
{
  static const std::string ret = detail::lookup_config_path();
  return ret;
}

}  // namespace kvikio
