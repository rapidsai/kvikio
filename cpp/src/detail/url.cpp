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
#include <stdexcept>
#include <utility>

#include <kvikio/detail/url.hpp>
#include <kvikio/error.hpp>
#include "curl/curl.h"

#define CHECK_CURL_URL_ERR(err_code) check_curl_url_err(err_code, __LINE__, __FILE__)

namespace kvikio::detail {
namespace {
inline void check_curl_url_err(CURLUcode err_code, int line_number, char const* filename)
{
  if (err_code == CURLUcode::CURLUE_OK) { return; }

  std::stringstream ss;
  ss << "KvikIO detects an URL error at: " << filename << ":" << line_number << ": ";
  char const* msg = curl_url_strerror(err_code);
  if (msg == nullptr) {
    ss << "(no message)";
  } else {
    ss << msg;
  }
  throw std::runtime_error(ss.str());
}
}  // namespace

CurlUrlHandle::CurlUrlHandle() : _handle(curl_url())
{
  KVIKIO_EXPECT(_handle != nullptr,
                "Libcurl is unable to allocate a URL handle (likely out of memory).");
}

CurlUrlHandle::~CurlUrlHandle()
{
  if (_handle) { curl_url_cleanup(_handle); }
}

CurlUrlHandle::CurlUrlHandle(CurlUrlHandle&& other) noexcept
  : _handle{std::exchange(other._handle, nullptr)}
{
}

CurlUrlHandle& CurlUrlHandle::operator=(CurlUrlHandle&& other) noexcept
{
  if (this != &other) {
    if (_handle) { curl_url_cleanup(_handle); }
    _handle = std::exchange(other._handle, nullptr);
  }

  return *this;
}

CURLU* CurlUrlHandle::get() const { return _handle; }

std::optional<std::string> UrlParser::extract_component(CurlUrlHandle& handle,
                                                        CURLUPart part,
                                                        unsigned int bitmask_component_flags,
                                                        std::optional<CURLUcode> exempt_err_code)
{
  char* value{};
  auto err_code = curl_url_get(handle.get(), part, &value, bitmask_component_flags);

  if (err_code == CURLUcode::CURLUE_OK && value != nullptr) {
    std::string result{value};
    curl_free(value);
    return result;
  }

  if (exempt_err_code.has_value() && exempt_err_code.value() == err_code) { return std::nullopt; }

  // Throws an exception and explains the reason.
  CHECK_CURL_URL_ERR(err_code);
  return std::nullopt;
}

UrlParser::UrlComponents UrlParser::parse(std::string const& url,
                                          std::optional<unsigned int> bitmask_url_flags,
                                          std::optional<unsigned int> bitmask_component_flags)
{
  if (!bitmask_url_flags.has_value()) { bitmask_url_flags = 0U; }
  if (!bitmask_component_flags) { bitmask_component_flags = 0U; }

  auto validate_non_empty_component = [](CurlUrlHandle handle) {

  };

  CurlUrlHandle handle;
  CHECK_CURL_URL_ERR(
    curl_url_set(handle.get(), CURLUPART_URL, url.c_str(), bitmask_url_flags.value()));

  UrlComponents components;
  CURLUcode err_code{};

  components.scheme = extract_component(
    handle, CURLUPART_SCHEME, bitmask_component_flags.value(), CURLUcode::CURLUE_NO_SCHEME);
  components.host = extract_component(
    handle, CURLUPART_HOST, bitmask_component_flags.value(), CURLUcode::CURLUE_NO_HOST);
  components.port = extract_component(
    handle, CURLUPART_PORT, bitmask_component_flags.value(), CURLUcode::CURLUE_NO_PORT);
  components.path  = extract_component(handle, CURLUPART_PATH, bitmask_component_flags.value());
  components.query = extract_component(
    handle, CURLUPART_QUERY, bitmask_component_flags.value(), CURLUcode::CURLUE_NO_QUERY);
  components.fragment = extract_component(
    handle, CURLUPART_FRAGMENT, bitmask_component_flags.value(), CURLUcode::CURLUE_NO_FRAGMENT);

  return components;
}
}  // namespace kvikio::detail
