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

#include <algorithm>
#include <stdexcept>

#include <kvikio/compat_mode.hpp>
#include <kvikio/detail/nvtx.hpp>
#include <kvikio/error.hpp>

namespace kvikio {

namespace detail {
CompatMode parse_compat_mode_str(std::string_view compat_mode_str)
{
  KVIKIO_NVTX_FUNC_RANGE();
  // Convert to lowercase
  std::string tmp{compat_mode_str};
  std::transform(
    tmp.begin(), tmp.end(), tmp.begin(), [](unsigned char c) { return std::tolower(c); });

  if (tmp == "on" || tmp == "true" || tmp == "yes" || tmp == "1") {
    return CompatMode::ON;
  } else if (tmp == "off" || tmp == "false" || tmp == "no" || tmp == "0") {
    return CompatMode::OFF;
  } else if (tmp == "auto") {
    return CompatMode::AUTO;
  } else {
    KVIKIO_FAIL("Unknown compatibility mode: " + std::string{tmp}, std::invalid_argument);
  }
  return {};
}

}  // namespace detail

}  // namespace kvikio
