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

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <kvikio/detail/url.hpp>
#include <stdexcept>

using ::testing::HasSubstr;
using ::testing::ThrowsMessage;

TEST(UrlTest, parse_scheme)
{
  {
    std::vector<std::string> invalid_scheme_urls{
      "invalid_scheme://host",
      // The S3 scheme is not supported by libcurl. Without the CURLU_NON_SUPPORT_SCHEME flag, an
      // exception is expected.
      "s3://host"};

    for (auto const& invalid_scheme_url : invalid_scheme_urls) {
      EXPECT_THAT([&] { kvikio::detail::UrlParser::parse(invalid_scheme_url); },
                  ThrowsMessage<std::runtime_error>(HasSubstr("KvikIO detects an URL error")));
    }
  }

  // With the CURLU_NON_SUPPORT_SCHEME flag, the S3 scheme is now accepted.
  {
    std::vector<std::string> schemes{"s3", "S3"};
    for (auto const& scheme : schemes) {
      auto parsed_url =
        kvikio::detail::UrlParser::parse(scheme + "://host", CURLU_NON_SUPPORT_SCHEME);
      EXPECT_EQ(parsed_url.scheme.value(), "s3");  // Lowercase due to CURL's normalization
    }
  }
}

TEST(UrlTest, parse_host)
{
  std::vector<std::string> invalid_host_urls{"http://host with spaces.com",
                                             "http://host[brackets].com",
                                             "http://host{braces}.com",
                                             "http://host<angle>.com",
                                             R"(http://host\backslash.com)",
                                             "http://host^caret.com",
                                             "http://host`backtick.com"};
  for (auto const& invalid_host_url : invalid_host_urls) {
    EXPECT_THROW({ kvikio::detail::UrlParser::parse(invalid_host_url); }, std::runtime_error);
  }
}
