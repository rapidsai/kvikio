/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <kvikio/detail/env.hpp>
#include <kvikio/detail/nvtx.hpp>
#include <kvikio/error.hpp>

namespace kvikio::detail {
std::string unwrap_or_env(std::optional<std::string> value,
                          std::string const& env_var,
                          std::optional<std::string> const& err_msg)
{
  KVIKIO_NVTX_FUNC_RANGE();
  if (value.has_value()) { return std::move(*value); }

  char const* env = std::getenv(env_var.c_str());
  if (env == nullptr) {
    if (!err_msg.has_value()) { return std::string(); }
    KVIKIO_FAIL(*err_msg, std::invalid_argument);
  }
  return std::string(env);
}
}  // namespace kvikio::detail