/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <stdexcept>

#include <kvikio/compat_mode.hpp>
#include <kvikio/detail/nvtx.hpp>
#include <kvikio/detail/utils.hpp>
#include <kvikio/error.hpp>

namespace kvikio {

namespace detail {
CompatMode parse_compat_mode_str(std::string_view compat_mode_str)
{
  KVIKIO_NVTX_FUNC_RANGE();
  auto const tmp = normalize_env_value(compat_mode_str);

  if (tmp == "on" || tmp == "true" || tmp == "yes" || tmp == "1") {
    return CompatMode::ON;
  } else if (tmp == "off" || tmp == "false" || tmp == "no" || tmp == "0") {
    return CompatMode::OFF;
  } else if (tmp == "auto") {
    return CompatMode::AUTO;
  } else {
    KVIKIO_FAIL("Unknown compatibility mode: " + std::string{tmp}, std::invalid_argument);
  }
}

}  // namespace detail

}  // namespace kvikio
