/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <regex>

#include <kvikio/detail/env.hpp>
#include <kvikio/detail/nvtx.hpp>
#include <kvikio/detail/remote_handle.hpp>
#include <kvikio/error.hpp>
#include <kvikio/hdfs.hpp>
#include <kvikio/remote_handle.hpp>
#include <kvikio/shim/libcurl.hpp>

namespace kvikio {

WebHdfsEndpoint::WebHdfsEndpoint(std::string url, std::optional<std::string> username)
  : RemoteEndpoint{RemoteEndpointType::WEBHDFS}, _username(std::move(username))
{
  // Extract two parts from the URL: components before (not including) the query and the query
  // itself
  std::regex static const url_pattern{R"(^([^?]+)\?([^#]*))"};
  // Regex meaning:
  // ^: From the start of the line
  // [^?]+: Matches non-question-mark characters one or more times. The question mark ushers in the
  // URL query component.
  // \?: Matches the question mark, which needs to be escaped.
  // [^#]*: Matches the non-pound characters zero or more times. The pound sign ushers in the URL
  // fragment component. It is very likely that this part does not exist.
  std::smatch url_match_results;
  bool found_query = std::regex_search(url, url_match_results, url_pattern);
  if (!found_query) {
    _url = url;
  } else {
    // URL components before (not including) the query
    _url = url_match_results[1].str();

    auto query = url_match_results[2].str();

    // Extract user name if provided. In WebHDFS, user name is specified as the key=value pair in
    // the query
    std::regex static const username_pattern{R"(user.name=([^&]+))"};
    // Regex meaning:
    // [^&]+: Matches the non-ampersand character one or more times. The ampersand delimits
    // different parameters.
    std::smatch username_match_results;
    bool found_username = std::regex_search(query, username_match_results, username_pattern);
    if (found_username) { _username = username_match_results[1].str(); }
  }

  // If the username is not specified by function parameter `username` or by the query string, check
  // the environment variable
  if (!_username.has_value()) {
    auto const* env_val = std::getenv("KVIKIO_WEBHDFS_USERNAME");
    if (env_val != nullptr) { _username = env_val; }
  }
}

WebHdfsEndpoint::WebHdfsEndpoint(std::string host,
                                 std::string port,
                                 std::string file_path,
                                 std::optional<std::string> username)
  : RemoteEndpoint{RemoteEndpointType::WEBHDFS}
{
  std::stringstream ss;
  ss << "http://" << host << ":" << port << "/webhdfs/v1" << file_path;
  _url      = ss.str();
  _username = detail::unwrap_or_env(std::move(username), "KVIKIO_WEBHDFS_USERNAME");
}

std::string WebHdfsEndpoint::str() const { return _url; }

void WebHdfsEndpoint::setopt(CurlHandle& curl)
{
  KVIKIO_NVTX_FUNC_RANGE();
  curl.setopt(CURLOPT_URL, _url.c_str());
  curl.setopt(CURLOPT_FOLLOWLOCATION, 1L);
}

std::size_t WebHdfsEndpoint::get_file_size()
{
  KVIKIO_NVTX_FUNC_RANGE();

  std::stringstream ss;
  ss << _url << "?";
  if (_username.has_value()) { ss << "user.name=" << _username.value() << "&"; }
  ss << "op=GETFILESTATUS";

  auto curl = create_curl_handle();
  curl.setopt(CURLOPT_URL, ss.str().c_str());
  curl.setopt(CURLOPT_FOLLOWLOCATION, 1L);

  std::string response;
  curl.setopt(CURLOPT_WRITEDATA, static_cast<void*>(&response));
  curl.setopt(CURLOPT_WRITEFUNCTION, detail::callback_get_string_response);

  curl.perform();

  long http_status_code{};
  curl.getinfo(CURLINFO_RESPONSE_CODE, &http_status_code);
  KVIKIO_EXPECT(http_status_code == 200, "HTTP response is not successful.");

  // The response is in JSON format. The file size is given by `"length":<file_size>`.
  std::regex static const pattern{R"("length"\s*:\s*(\d+)[^\d])"};
  // Regex meaning:
  // \s*: Matches the space character zero or more times.
  // \d+: Matches the digit one or more times.
  // [^\d]: Matches a non-digit character.
  std::smatch match_results;
  bool found = std::regex_search(response, match_results, pattern);
  KVIKIO_EXPECT(
    found, "Regular expression search failed. Cannot extract file length from the JSON response.");
  return std::stoull(match_results[1].str());
}

void WebHdfsEndpoint::setup_range_request(CurlHandle& curl,
                                          std::size_t file_offset,
                                          std::size_t size)
{
  // WebHDFS does not support CURLOPT_RANGE. The range is specified as query parameters in the URL.
  KVIKIO_NVTX_FUNC_RANGE();
  std::stringstream ss;
  ss << _url << "?";
  if (_username.has_value()) { ss << "user.name=" << _username.value() << "&"; }
  ss << "op=OPEN&offset=" << file_offset << "&length=" << size;
  curl.setopt(CURLOPT_URL, ss.str().c_str());
}

bool WebHdfsEndpoint::is_url_valid(std::string const& url) noexcept
{
  try {
    std::regex static const pattern(R"(^https?://[^/]+:\d+/webhdfs/v1/.+$)",
                                    std::regex_constants::icase);
    return std::regex_match(url, pattern);
  } catch (...) {
    return false;
  }
}
}  // namespace kvikio
