/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <array>
#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <utility>

#include <curl/curl.h>
#include <kvikio/detail/url.hpp>
#include <kvikio/error.hpp>

#define CHECK_CURL_URL_ERR(err_code) check_curl_url_err(err_code, __LINE__, __FILE__)

namespace kvikio::detail {
namespace {
void check_curl_url_err(CURLUcode err_code, int line_number, char const* filename)
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

CurlUrlHandle::~CurlUrlHandle() noexcept
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

std::optional<std::string> UrlParser::extract_component(
  CurlUrlHandle const& handle,
  CURLUPart part,
  std::optional<unsigned int> bitmask_component_flags,
  std::optional<CURLUcode> allowed_err_code)
{
  if (!bitmask_component_flags.has_value()) { bitmask_component_flags = 0U; }

  char* value{};
  auto err_code = curl_url_get(handle.get(), part, &value, bitmask_component_flags.value());

  if (err_code == CURLUcode::CURLUE_OK && value != nullptr) {
    std::string result{value};
    curl_free(value);
    return result;
  }

  if (allowed_err_code.has_value() && allowed_err_code.value() == err_code) { return std::nullopt; }

  // Throws an exception and explains the reason.
  CHECK_CURL_URL_ERR(err_code);
  return std::nullopt;
}

std::optional<std::string> UrlParser::extract_component(
  std::string const& url,
  CURLUPart part,
  std::optional<unsigned int> bitmask_url_flags,
  std::optional<unsigned int> bitmask_component_flags,
  std::optional<CURLUcode> allowed_err_code)
{
  if (!bitmask_url_flags.has_value()) { bitmask_url_flags = 0U; }
  if (!bitmask_component_flags.has_value()) { bitmask_component_flags = 0U; }

  CurlUrlHandle handle;
  CHECK_CURL_URL_ERR(
    curl_url_set(handle.get(), CURLUPART_URL, url.c_str(), bitmask_url_flags.value()));

  return extract_component(handle, part, bitmask_component_flags, allowed_err_code);
}

UrlParser::UrlComponents UrlParser::parse(std::string const& url,
                                          std::optional<unsigned int> bitmask_url_flags,
                                          std::optional<unsigned int> bitmask_component_flags)
{
  if (!bitmask_url_flags.has_value()) { bitmask_url_flags = 0U; }
  if (!bitmask_component_flags.has_value()) { bitmask_component_flags = 0U; }

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

UrlBuilder::UrlBuilder() {}

UrlBuilder::UrlBuilder(std::string const& url, std::optional<unsigned int> bitmask_url_flags)
{
  if (!bitmask_url_flags.has_value()) { bitmask_url_flags = 0U; }

  CHECK_CURL_URL_ERR(
    curl_url_set(_handle.get(), CURLUPART_URL, url.c_str(), bitmask_url_flags.value()));
}

UrlBuilder::UrlBuilder(UrlParser::UrlComponents const& components,
                       std::optional<unsigned int> bitmask_url_flags)
{
  // Start with an empty builder
  // Set each component if present
  if (components.scheme.has_value()) { set_scheme(components.scheme); }
  if (components.host.has_value()) { set_host(components.host); }
  if (components.port.has_value()) { set_port(components.port); }
  if (components.path.has_value()) { set_path(components.path); }
  if (components.query.has_value()) { set_query(components.query); }
  if (components.fragment.has_value()) { set_fragment(components.fragment); }
}

UrlBuilder& UrlBuilder::set_component(CURLUPart part,
                                      char const* value,
                                      std::optional<unsigned int> flags)
{
  if (!flags.has_value()) { flags = 0U; }

  CHECK_CURL_URL_ERR(curl_url_set(_handle.get(), part, value, flags.value()));
  return *this;
}

UrlBuilder& UrlBuilder::set_scheme(std::optional<std::string> const& scheme)
{
  auto const* value = scheme.has_value() ? scheme.value().c_str() : nullptr;
  return set_component(CURLUPART_SCHEME, value);
}

UrlBuilder& UrlBuilder::set_host(std::optional<std::string> const& host)
{
  auto const* value = host.has_value() ? host.value().c_str() : nullptr;
  return set_component(CURLUPART_HOST, value);
}

UrlBuilder& UrlBuilder::set_port(std::optional<std::string> const& port)
{
  auto const* value = port.has_value() ? port.value().c_str() : nullptr;
  return set_component(CURLUPART_PORT, value);
}

UrlBuilder& UrlBuilder::set_path(std::optional<std::string> const& path)
{
  auto const* value = path.has_value() ? path.value().c_str() : nullptr;
  return set_component(CURLUPART_PATH, value);
}

UrlBuilder& UrlBuilder::set_query(std::optional<std::string> const& query)
{
  auto const* value = query.has_value() ? query.value().c_str() : nullptr;
  return set_component(CURLUPART_QUERY, value);
}

UrlBuilder& UrlBuilder::set_fragment(std::optional<std::string> const& fragment)
{
  auto const* value = fragment.has_value() ? fragment.value().c_str() : nullptr;
  return set_component(CURLUPART_FRAGMENT, value);
}

std::string UrlBuilder::build(std::optional<unsigned int> bitmask_component_flags) const
{
  if (!bitmask_component_flags.has_value()) { bitmask_component_flags = 0U; }

  char* url = nullptr;
  CHECK_CURL_URL_ERR(
    curl_url_get(_handle.get(), CURLUPART_URL, &url, bitmask_component_flags.value()));

  KVIKIO_EXPECT(
    url != nullptr, "Failed to build URL: curl_url_get returned nullptr", std::runtime_error);

  std::string result(url);
  curl_free(url);
  return result;
}

std::string UrlBuilder::build_manually(UrlParser::UrlComponents const& components)
{
  std::string url;
  if (components.scheme) { url += components.scheme.value() + "://"; }
  if (components.host) { url += components.host.value(); }
  if (components.port) { url += ":" + components.port.value(); }
  if (components.path) { url += components.path.value(); }
  if (components.query) { url += "?" + components.query.value(); }
  if (components.fragment) { url += "#" + components.fragment.value(); }
  return url;
}

namespace {
/**
 * @brief Compile-time encoding lookup table
 *
 * ASCII characters will be percent-encoded. For example, = has a hexadecimal value of 3D, and the
 * encoding result is %3D. Characters outside the ASCII region are encoded to NUL and map to an
 * empty std::string.
 */
struct EncodingTable {
  std::array<unsigned char[4], 256> table;
  constexpr EncodingTable() : table{}
  {
    char const num_to_chars[] = "0123456789ABCDEF";
    for (uint16_t idx = 0U; idx < table.size(); ++idx) {
      if (idx < 128) {
        table[idx][0] = '%';
        table[idx][1] = num_to_chars[idx >> 4];
        table[idx][2] = num_to_chars[idx & 0x0F];
        table[idx][3] = '\0';
      } else {
        table[idx][0] = '\0';
      }
    }
  }
};
}  // namespace

std::string UrlEncoder::encode_path(std::string_view path, std::string_view chars_to_encode)
{
  constexpr EncodingTable encoding_table{};

  std::array<bool, 256> should_encode{};
  for (auto const c : chars_to_encode) {
    std::size_t idx    = static_cast<unsigned char>(c);
    should_encode[idx] = true;
  }

  std::string result;
  for (auto const c : path) {
    std::size_t idx = static_cast<unsigned char>(c);
    if (should_encode[idx]) {
      // If the character is within chars_to_encode, encode it
      result += std::string{reinterpret_cast<char const*>(encoding_table.table[idx])};
    } else {
      // Otherwise, pass it through
      result += c;
    }
  }

  return result;
}

}  // namespace kvikio::detail
