/*
 * Copyright (c) 2024-2025, NVIDIA CORPORATION.
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
    std::vector<std::string> inputs{
      "ON", "on", "On", "TRUE", "true", "True", "YES", "yes", "Yes", "1"};
    for (const auto& input : inputs) {
      EXPECT_EQ(kvikio::detail::parse_compat_mode_str(input), kvikio::CompatMode::ON);
    }
  }

  {
    std::vector<std::string> inputs{
      "OFF", "off", "oFf", "FALSE", "false", "False", "NO", "no", "No", "0"};
    for (const auto& input : inputs) {
      EXPECT_EQ(kvikio::detail::parse_compat_mode_str(input), kvikio::CompatMode::OFF);
    }
  }

  {
    std::vector<std::string> inputs{"AUTO", "auto", "aUtO"};
    for (const auto& input : inputs) {
      EXPECT_EQ(kvikio::detail::parse_compat_mode_str(input), kvikio::CompatMode::AUTO);
    }
  }

  {
    std::vector<std::string> inputs{"", "invalidOption", "11", "*&^Yes"};
    for (const auto& input : inputs) {
      EXPECT_THROW(kvikio::detail::parse_compat_mode_str(input), std::invalid_argument);
    }
  }
}

TEST(Defaults, parse_http_status_codes)
{
  {
    std::vector<std::string> inputs{
      "429,500", "429, 500", " 429,500", "429,  500", "429 ,500", "429,500 "};
    std::vector<int> expected = {429, 500};
    // std::string const input = "429,500,501,503";
    for (const auto& input : inputs) {
      EXPECT_EQ(kvikio::detail::parse_http_status_codes("KVIKIO_HTTP_STATUS_CODES", input),
                expected);
    }
  }

  {
    std::vector<std::string> inputs{"429,", ",429", "a,b", "429,,500", "429,1000"};
    for (const auto& input : inputs) {
      EXPECT_THROW(kvikio::detail::parse_http_status_codes("KVIKIO_HTTP_STATUS_CODES", input),
                   std::invalid_argument);
    }
  }
}
