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

#include <regex>

#include <kvikio/detail/remote_handle.hpp>
#include <kvikio/detail/url.hpp>
#include <kvikio/error.hpp>
#include <kvikio/hdfs.hpp>
#include <kvikio/nvtx.hpp>
#include <kvikio/remote_handle.hpp>
#include <kvikio/shim/libcurl.hpp>

namespace kvikio {

namespace {
/**
 * @brief
 *
 * @param url
 * @return std::string
 */
std::string url_before_query(std::string const& url)
{
  std::regex const pattern{R"(^([^?]+)\?)"};
  std::smatch match_results;
  bool found = std::regex_search(url, match_results, pattern);
  if (found) { return match_results[1].str(); }
  return url;
}

}  // namespace

WebHdfsEndpoint::WebHdfsEndpoint(std::string url)
{
  // Extract user name if provided
  // In WebHDFS, user name is specified as the first key=param pair in the query
  detail::Url url_obj{url};

  auto query = url_obj.query();
  if (query.has_value()) {
    std::regex const pattern{R"(^user.name=[^&]+?)"};
    std::smatch match_results;
    if (std::regex_search(query.value(), match_results, pattern)) {
      _username = match_results[1].str();
    }
  }

  _url = url_before_query(url);
}

WebHdfsEndpoint::WebHdfsEndpoint(std::string host,
                                 std::string port,
                                 std::string file_path,
                                 std::optional<std::string> username)
{
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
  std::string const url_full = _url + std::string{"?user.name=rladmin&op=GETFILESTATUS"};

  auto curl = create_curl_handle();
  curl.setopt(CURLOPT_URL, url_full.c_str());
  curl.setopt(CURLOPT_FOLLOWLOCATION, 1L);

  std::string response;
  curl.setopt(CURLOPT_WRITEDATA, static_cast<void*>(&response));
  curl.setopt(CURLOPT_WRITEFUNCTION, detail::callback_get_response);

  curl.perform();

  long http_status_code{};
  curl.getinfo(CURLINFO_RESPONSE_CODE, &http_status_code);
  KVIKIO_EXPECT(http_status_code == 200, "HTTP response is not successful.");

  std::regex const pattern{R"("length"\s*:\s*(\d+)[^\d])"};
  std::smatch match_results;
  bool found = std::regex_search(response, match_results, pattern);
  KVIKIO_EXPECT(
    found, "Regular expression search failed. Cannot extract file length from the JSON response.");
  return std::stoi(match_results[1].str());
}

void WebHdfsEndpoint::setup_range_request(CurlHandle& curl,
                                          std::size_t file_offset,
                                          std::size_t size)
{
  KVIKIO_NVTX_FUNC_RANGE();
  std::stringstream ss;
  ss << _url << "?user.name=rladmin&op=OPEN&offset=" << file_offset << "&length=" << size;
  std::string const url_full = ss.str();
  curl.setopt(CURLOPT_URL, url_full.c_str());
}
}  // namespace kvikio
