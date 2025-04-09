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

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace kvikio {
namespace detail {
/**
 * @brief Parse a string of comma-separated string of HTTP status codes.
 *
 * @param env_var_name The environment variable holding the string.
 * Used to report errors.
 * @param status_codes The comma-separated string of HTTP status
 * codes. Each code should be a 3-digit integer.
 *
 * @return The vector with the parsed, integer HTTP status codes.
 */
std::vector<int> parse_http_status_codes(std::string_view env_var_name,
                                         std::string const& status_codes);
}  // namespace detail

}  // namespace kvikio
