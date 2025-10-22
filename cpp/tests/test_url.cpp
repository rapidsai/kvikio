/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <kvikio/detail/url.hpp>
#include <kvikio/error.hpp>
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

TEST(UrlTest, build_url)
{
  // Build a URL from scratch
  {
    std::string scheme{"https"};
    std::string host{"api.example.com"};
    std::string port{"8080"};
    std::string path{"/v1/users"};
    std::string query{"page=1&limit=10"};
    std::string fragment{"results"};
    std::stringstream ss;
    ss << scheme << "://" << host << ":" << port << path << "?" << query << "#" << fragment;

    {
      auto url = kvikio::detail::UrlBuilder()
                   .set_scheme("https")
                   .set_host("api.example.com")
                   .set_port("8080")
                   .set_path("/v1/users")
                   .set_query("page=1&limit=10")
                   .set_fragment("results")
                   .build();

      EXPECT_EQ(url, ss.str());
    }

    // The components do not have to be specified in their correct order
    {
      auto url = kvikio::detail::UrlBuilder()
                   .set_fragment("results")
                   .set_scheme("https")
                   .set_path("/v1/users")
                   .set_host("api.example.com")
                   .set_query("page=1&limit=10")
                   .set_port("8080")
                   .build();

      EXPECT_EQ(url, ss.str());
    }
  }

  // Modify an existing URL
  {
    std::string scheme_host{"https://api.example.com"};
    std::string query{"page=1&limit=10"};

    std::string old_path{"/old/path/file.txt"};
    std::string new_path{"/new/path/document.pdf"};

    // Modify the path
    {
      std::string old_url          = scheme_host + old_path + "?" + query;
      std::string expected_new_url = scheme_host + new_path + "?" + query;

      auto actual_new_url = kvikio::detail::UrlBuilder(old_url).set_path(new_path).build();
      EXPECT_EQ(actual_new_url, expected_new_url);
    }

    // Modify the path and add the query
    std::string port{"8080"};
    std::string old_url          = scheme_host + old_path;
    std::string expected_new_url = scheme_host + ":" + port + new_path + "?" + query;

    auto actual_new_url = kvikio::detail::UrlBuilder(old_url)
                            .set_port(port)
                            .set_path(new_path)
                            .set_query(query)
                            .build();
    EXPECT_EQ(actual_new_url, expected_new_url);
  }

  // Build from parsed components
  {
    std::string scheme{"https"};
    std::string host{"api.example.com"};
    std::string path{"/v1/users"};
    std::string query{"page=1&limit=10"};
    std::stringstream ss;
    ss << scheme << "://" << host << path << "?" << query;

    // First parse an existing URL
    auto components = kvikio::detail::UrlParser::parse(ss.str());

    // Modify components
    components.path = "/v2/api";
    components.port = "443";

    // Build new URL from modified components
    auto actual_new_url = kvikio::detail::UrlBuilder(components).build();

    // Expected URL
    ss.str("");
    ss << scheme << "://" << host << ":" << components.port.value() << components.path.value()
       << "?" << query;

    EXPECT_EQ(actual_new_url, ss.str());
  }

  // AWS S3-like URL
  {
    std::string path = "/my-bucket/&$@=;:+,.txt";
    auto url = kvikio::detail::UrlBuilder("https://s3.region.amazonaws.com").set_path(path).build();
    std::string encoded_path = kvikio::detail::UrlEncoder::encode_path(path);

    auto actual_encoded_url = kvikio::detail::UrlBuilder(url).set_path(encoded_path).build();
    std::string expected_encoded_url{
      "https://s3.region.amazonaws.com/my-bucket/%26%24%40%3D%3B%3A%2B%2C.txt"};

    std::transform(actual_encoded_url.begin(),
                   actual_encoded_url.end(),
                   actual_encoded_url.begin(),
                   [](unsigned char c) { return std::tolower(c); });

    std::transform(expected_encoded_url.begin(),
                   expected_encoded_url.end(),
                   expected_encoded_url.begin(),
                   [](unsigned char c) { return std::tolower(c); });

    EXPECT_EQ(actual_encoded_url, expected_encoded_url);
  }
}

TEST(UrlTest, encoding_table)
{
  // Look up the reserved characters (RFC 3986 section 2.2) in the encoding table
  {
    std::string special_chars{"!#$&\'()*+,/:;=?@[]"};
    std::string expected_result{"%21%23%24%26%27%28%29%2A%2B%2C%2F%3A%3B%3D%3F%40%5B%5D"};
    // First parameter: string containing special characters
    // Second parameter: a sequence of special characters to be encoded
    std::string actual_result =
      kvikio::detail::UrlEncoder::encode_path(special_chars, special_chars);
    EXPECT_EQ(actual_result, expected_result);
  }

  // Check a few samples from the encoding table. Out-of-bound characters (beyond ASCII table) are
  // expected to be encoded to empty strings.
  {
    std::array<unsigned char, 6> input{0,  // First ASCII char NUL
                                       '\x3D',
                                       127,  // Last ASCII char DEL
                                       128,  // Out-of-bound chars
                                       200,
                                       255};
    std::array<std::string, sizeof(input)> expected_results{"%00",
                                                            "%3D",
                                                            "%7F"
                                                            "",
                                                            "",
                                                            ""};
    for (std::size_t i = 0; i < input.size(); ++i) {
      std::string s{static_cast<char>(input[i])};
      std::string actual_result = kvikio::detail::UrlEncoder::encode_path(s, s);
      EXPECT_EQ(actual_result, expected_results[i]);
    }
  }

  // Check control characters
  {
    std::map<char, std::string> mapping{
      {'\x00', "%00"}, {'\x1A', "%1A"}, {'\x1F', "%1F"}, {'\x7F', "%7F"}};

    for (auto const [question, answer] : mapping) {
      // Construct a string view for the character, and specify the size explicitly to take account
      // of NUL
      std::string sv{&question, 1};
      std::string result = kvikio::detail::UrlEncoder::encode_path(sv, sv);
      EXPECT_EQ(result, answer);
    }
  }

  // Check out-of-bound characters
  {
    unsigned char out_of_bound_chars[] = {128, 200, 255};
    std::string_view sv{reinterpret_cast<char*>(out_of_bound_chars), sizeof(out_of_bound_chars)};
    std::string result = kvikio::detail::UrlEncoder::encode_path(sv, sv);
    EXPECT_EQ(result, "");
  }
}

TEST(UrlTest, encode_url)
{
  // Path does not contain characters that require special handling, so no character is encoded
  {
    std::string original{"abc123/-_..bin"};
    auto encoded = kvikio::detail::UrlEncoder::encode_path(original);
    EXPECT_EQ(original, encoded);
  }

  // chars_to_encode is empty, so no character is encoded
  {
    std::string original{"abc123/!-_.*'()/&$@=;:+ ,?.bin"};
    auto encoded = kvikio::detail::UrlEncoder::encode_path(original, {});
    EXPECT_EQ(original, encoded);
  }

  // Test all characters mentioned by AWS documentation that require special handling
  {
    std::string const& input{kvikio::detail::UrlEncoder::aws_special_chars};
    auto encoded = kvikio::detail::UrlEncoder::encode_path(input);

    // Encoding is performed, so the string is expected to be changed
    EXPECT_NE(input, encoded);

    auto* curl     = curl_easy_init();
    auto* expected = curl_easy_escape(curl, input.data(), input.size());
    EXPECT_NE(expected, nullptr);
    EXPECT_EQ(encoded, std::string{expected});

    curl_free(expected);
    curl_easy_cleanup(curl);

    // aws_special_chars does not contain %, so double encoding is expected to not alter anything
    auto double_encoded = kvikio::detail::UrlEncoder::encode_path(encoded);
    EXPECT_EQ(encoded, double_encoded);
  }
}
