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
 * Resolution order:
 * - If `value` has a value, return it
 * - If environment variable `env_var` is set, return its value (even if empty)
 * - Return std::nullopt if err_msg is std::nullopt; throw an exception otherwise
 *
 * @param value The value to unwrap.
 * @param env_var The name of the environment variable to check if `value` isn't set.
 * @param err_msg Optional error message that controls whether to throw an exception if neither
 * source provides a value.
 * @return The resolved value, or std::nullopt if neither source provides a value.
 */
std::optional<std::string> unwrap_or_env(std::optional<std::string> value,
                                         std::string const& env_var,
                                         std::optional<std::string> const& err_msg = std::nullopt);
}  // namespace kvikio::detail
