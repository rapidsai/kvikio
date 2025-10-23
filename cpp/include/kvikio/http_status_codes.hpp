/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
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
