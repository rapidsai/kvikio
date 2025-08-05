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

#include <optional>
#include <sstream>

#include <kvikio/shim/libcurl.hpp>

#define CHECK_URL_ERR(err_code)                                  \
  do {                                                           \
    kvikio::detail::check_url_err(err_code, __LINE__, __FILE__); \
  } while (0)

namespace kvikio::detail {

inline void check_url_err(CURLUcode err_code, int line_number, char const* filename)
{
  if (err_code == CURLUcode::CURLUE_OK) { return; }
  auto* msg = curl_url_strerror(err_code);
  std::stringstream ss;
  ss << "KvikIO URL parse failure: " << msg << " at " << filename << ":" << line_number << ": ";
  throw std::runtime_error(ss.str());
}

class Url {
 private:
  std::string _url;
  std::optional<std::string> _scheme;
  std::optional<std::string> _host;
  std::optional<std::string> _port;
  std::optional<std::string> _file_path;
  std::optional<std::string> _query;

 public:
  Url(std::string const& url);

  std::optional<std::string> const& scheme();
  std::optional<std::string> const& host();
  std::optional<std::string> const& file_path();
  std::optional<std::string> const& port();
  std::optional<std::string> const& query();
};

}  // namespace kvikio::detail
