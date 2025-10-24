/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <optional>
#include <string>

namespace kvikio::detail {
/**
 * @brief Unwrap an optional parameter, obtaining a fallback from the environment.
 *
 * If not nullopt, the optional's value is returned. Otherwise, the environment
 * variable `env_var` is used. If that also doesn't have a value:
 *   - if `err_msg` is nullopt, an empty string is returned.
 *   - if `err_msg` has a value, `std::invalid_argument(err_msg.value())` is thrown.
 *
 * @param value The value to unwrap.
 * @param env_var The name of the environment variable to check if `value` isn't set.
 * @param err_msg Optional error message to throw on error, or nullopt to return empty string.
 * @return The unwrapped value, environment variable value, or empty string.
 */
std::string unwrap_or_env(std::optional<std::string> value,
                          std::string const& env_var,
                          std::optional<std::string> const& err_msg = std::nullopt);
}  // namespace kvikio::detail