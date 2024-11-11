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

#include <stdexcept>

#include <gtest/gtest.h>
#include <kvikio/defaults.hpp>

TEST(Defaults, parse_compat_mode_str)
{
  {
    std::vector<std::string> inputs{"ON", "on", "On"};
    for (const auto& input : inputs) {
      EXPECT_EQ(kvikio::detail::parse_compat_mode_str(input), kvikio::detail::CompatMode::ON);
    }
  }

  {
    std::vector<std::string> inputs{"OFF", "off", "oFf"};
    for (const auto& input : inputs) {
      EXPECT_EQ(kvikio::detail::parse_compat_mode_str(input), kvikio::detail::CompatMode::OFF);
    }
  }

  {
    std::vector<std::string> inputs{"AUTO", "auto", "aUtO"};
    for (const auto& input : inputs) {
      EXPECT_EQ(kvikio::detail::parse_compat_mode_str(input), kvikio::detail::CompatMode::AUTO);
    }
  }

  {
    std::vector<std::string> inputs{"", "invalidOption"};
    for (const auto& input : inputs) {
      EXPECT_THROW(kvikio::detail::parse_compat_mode_str(input), std::invalid_argument);
    }
  }
}
