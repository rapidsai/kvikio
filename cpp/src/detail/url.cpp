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

#include <functional>
#include <memory>
#include <optional>

#include <curl/urlapi.h>
#include <kvikio/detail/url.hpp>
#include <kvikio/error.hpp>
#include <kvikio/shim/libcurl.hpp>

namespace kvikio::detail {

using UrlHandle = std::unique_ptr<CURLU, std::function<decltype(curl_url_cleanup)>>;

class UrlPart {
 private:
  char* _part_raw_str{};
  std::optional<std::string> _part_str{};

 public:
  UrlPart(CURLUPart part,
          UrlHandle const& handle,
          std::optional<CURLUcode> absence_err_code = std::nullopt,
          unsigned int flags                        = 0)
  {
    auto err_code = curl_url_get(handle.get(), part, &_part_raw_str, flags);
    if (err_code == CURLUcode::CURLUE_OK) {
      // _part_str holds a copy of the string content in _part_raw_str.
      _part_str = _part_raw_str;
    } else if (absence_err_code.has_value() && (err_code == absence_err_code.value())) {
      // In case of error, the state of _part_raw_str is unspecified, so we manually reset.
      _part_raw_str = nullptr;
    } else {
      CHECK_URL_ERR(err_code);
    }
  };

  ~UrlPart() noexcept
  {
    if (_part_raw_str != nullptr) { curl_free(_part_raw_str); }
  }

  std::optional<std::string> const& get() const noexcept { return _part_str; }
};

Url::Url(std::string const& url) : _url{url}
{
  UrlHandle url_handle{curl_url(), curl_url_cleanup};

  CHECK_URL_ERR(curl_url_set(url_handle.get(), CURLUPART_URL, _url.c_str(), 0));

  UrlPart scheme{CURLUPART_SCHEME, url_handle, CURLUE_NO_SCHEME};
  UrlPart host{CURLUPART_HOST, url_handle, CURLUE_NO_HOST};

  // If the URL does not contain a file path, it defaults to "/". This is different from other URL
  // components.
  UrlPart file_path{CURLUPART_PATH, url_handle};

  UrlPart port{CURLUPART_PORT, url_handle, CURLUE_NO_PORT};
  UrlPart query{CURLUPART_QUERY, url_handle, CURLUE_NO_QUERY};

  _scheme    = scheme.get();
  _host      = host.get();
  _file_path = file_path.get();
  _port      = port.get();
  _query     = query.get();
}

std::optional<std::string> const& Url::scheme() { return _scheme; }

std::optional<std::string> const& Url::host() { return _host; }

std::optional<std::string> const& Url::file_path() { return _file_path; }

std::optional<std::string> const& Url::port() { return _port; }

std::optional<std::string> const& Url::query() { return _query; }

}  // namespace kvikio::detail
