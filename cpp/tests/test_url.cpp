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

#include <sstream>

#include <gtest/gtest.h>
#include <kvikio/detail/url.hpp>
#include <stdexcept>

TEST(UrlTest, valid_url)
{
  std::string const scheme{"http"};
  std::string const host{"www.example.com"};
  std::string const port{"1234"};
  std::string const file_path{"/home/sample-user/tmp"};
  std::string const query{"param1=value1&param2=value2"};
  std::string const fragment{"section"};

  // URL has all the components
  {
    std::stringstream ss;
    ss << scheme << "://" << host << ":" << port << file_path << "?" << query << "#" << fragment;
    kvikio::detail::Url url{ss.str()};
    EXPECT_EQ(url.scheme().value(), scheme);
    EXPECT_EQ(url.host().value(), host);
    EXPECT_EQ(url.file_path().value(), file_path);
    EXPECT_EQ(url.port().value(), port);
    EXPECT_EQ(url.query().value(), query);
  }

  // Port, query and fragment are missing
  {
    std::stringstream ss;
    ss << scheme << "://" << host;
    kvikio::detail::Url url{ss.str()};
    EXPECT_EQ(url.scheme().value(), scheme);
    EXPECT_EQ(url.host().value(), host);
    // Path defaults to "/" when not specified in URL
    EXPECT_EQ(url.file_path().value(), "/");
    EXPECT_FALSE(url.port().has_value());
    EXPECT_FALSE(url.query().has_value());
  }
}

TEST(UrlTest, invalid_url)
{
  // URL is empty
  {
    EXPECT_THROW({ kvikio::detail::Url url{""}; }, std::runtime_error);
  }

  // Invalid scheme
  {
    EXPECT_THROW({ kvikio::detail::Url url{"invalid-scheme://"}; }, std::runtime_error);
  }

  // Invalid host
  {
    // Space character is not allowed
    EXPECT_THROW({ kvikio::detail::Url url{"example .com"}; }, std::runtime_error);
    // @ is not allowed
    EXPECT_THROW({ kvikio::detail::Url url{"example@com"}; }, std::runtime_error);
    // Leading dash is not allowed
    EXPECT_THROW({ kvikio::detail::Url url{"-example.com"}; }, std::runtime_error);
  }

  // Invalid port
  {
    // Port range is exceeded
    EXPECT_THROW({ kvikio::detail::Url url{"http://example.com:65536"}; }, std::runtime_error);
    // Dot is not allowed
    EXPECT_THROW({ kvikio::detail::Url url{"http://example.com:12.34"}; }, std::runtime_error);
    // Port is not numeric
    EXPECT_THROW({ kvikio::detail::Url url{"http://example.com:1234a"}; }, std::runtime_error);
  }
}
