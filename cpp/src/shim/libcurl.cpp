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

#include <cstring>
#include <functional>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <curl/curl.h>

#include <kvikio/defaults.hpp>
#include <kvikio/error.hpp>
#include <kvikio/parallel_operation.hpp>
#include <kvikio/posix_io.hpp>
#include <kvikio/shim/libcurl.hpp>
#include <kvikio/utils.hpp>

namespace kvikio {

LibCurl::LibCurl()
{
  CURLcode err = curl_global_init(CURL_GLOBAL_DEFAULT);
  if (err != CURLE_OK) {
    throw std::runtime_error("cannot initialize libcurl - errorcode: " + std::to_string(err));
  }
  curl_version_info_data* ver = curl_version_info(::CURLVERSION_NOW);
  if ((ver->features & CURL_VERSION_THREADSAFE) == 0) {
    throw std::runtime_error("cannot initialize libcurl - built with thread safety disabled");
  }
}

LibCurl::~LibCurl() noexcept
{
  _free_curl_handles.clear();
  curl_global_cleanup();
}

LibCurl& LibCurl::instance()
{
  static LibCurl _instance;
  return _instance;
}

LibCurl::UniqueHandlePtr LibCurl::get_free_handle()
{
  UniqueHandlePtr ret;
  std::lock_guard const lock(_mutex);
  if (!_free_curl_handles.empty()) {
    ret = std::move(_free_curl_handles.back());
    _free_curl_handles.pop_back();
  }
  return ret;
}

LibCurl::UniqueHandlePtr LibCurl::get_handle()
{
  // Check if we have a free handle available.
  UniqueHandlePtr ret = get_free_handle();
  if (ret) {
    curl_easy_reset(ret.get());
  } else {
    // If not, we create a new handle.
    CURL* raw_handle = curl_easy_init();
    if (raw_handle == nullptr) {
      throw std::runtime_error("libcurl: call to curl_easy_init() failed");
    }
    ret = UniqueHandlePtr(raw_handle, curl_easy_cleanup);
  }
  return ret;
}

void LibCurl::retain_handle(UniqueHandlePtr handle)
{
  std::lock_guard const lock(_mutex);
  _free_curl_handles.push_back(std::move(handle));
}

CurlHandle::CurlHandle(LibCurl::UniqueHandlePtr handle,
                       std::string source_file,
                       std::string source_line)
  : _handle{std::move(handle)},
    _source_file(std::move(source_file)),
    _source_line(std::move(source_line))
{
  // Need CURLOPT_NOSIGNAL to support threading, see
  // <https://curl.se/libcurl/c/CURLOPT_NOSIGNAL.html>
  setopt(CURLOPT_NOSIGNAL, 1L);

  // We always set CURLOPT_ERRORBUFFER to get better error messages.
  _errbuf[0] = 0;  // Set the error buffer as empty.
  setopt(CURLOPT_ERRORBUFFER, _errbuf);

  // Make curl_easy_perform() fail when receiving HTTP code errors.
  setopt(CURLOPT_FAILONERROR, 1L);
}

CurlHandle::~CurlHandle() noexcept { LibCurl::instance().retain_handle(std::move(_handle)); }

CURL* CurlHandle::handle() noexcept { return _handle.get(); }

void CurlHandle::perform()
{
  // Perform the curl operation and check for errors.
  CURLcode err = curl_easy_perform(handle());
  if (err != CURLE_OK) {
    std::string msg(_errbuf);  // We can do this because we always initialize `_errbuf` as empty.
    std::stringstream ss;
    ss << "curl_easy_perform() error near " << _source_file << ":" << _source_line;
    if (msg.empty()) {
      ss << "(" << curl_easy_strerror(err) << ")";
    } else {
      ss << "(" << msg << ")";
    }
    throw std::runtime_error(ss.str());
  }
}

}  // namespace kvikio
