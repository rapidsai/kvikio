/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <chrono>
#include <cstddef>
#include <format>
#include <optional>
#include <regex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <kvikio/azure.hpp>
#include <kvikio/defaults.hpp>
#include <kvikio/detail/azure_shared_key.hpp>
#include <kvikio/detail/crypto.hpp>
#include <kvikio/detail/env.hpp>
#include <kvikio/detail/nvtx.hpp>
#include <kvikio/detail/remote_handle.hpp>
#include <kvikio/detail/url.hpp>
#include <kvikio/error.hpp>
#include <kvikio/remote_handle.hpp>
#include <kvikio/shim/libcurl.hpp>
#include <kvikio/statistics/counters.hpp>

namespace kvikio {

namespace {

/// The parts of an Azure Blob URL that the Azure Blob endpoints need.
struct BlobUrl {
  std::string encoded_url;
  std::string account;
  std::string encoded_path;
};

/**
 * @brief Parse an Azure Blob URL.
 *
 * @param url A URL.
 * @return The parts of `url`, or std::nullopt if `url` does not have the form
 * `https://<account>.blob.core.windows.net/<container>/<blob>` or if it has a query.
 */
std::optional<BlobUrl> parse_blob_url(std::string const& url)
{
  detail::UrlParser::UrlComponents components;
  try {
    components = detail::UrlParser::parse(url);
  } catch (std::runtime_error const&) {
    // libcurl cannot parse `url` at all.
    return std::nullopt;
  }
  // A URL with a query is rejected. `detail::azure_shared_key_headers()` cannot sign a query, and
  // `AzureBlobOAuthEndpoint` follows the same rule. Therefore a URL that works with one endpoint
  // also works with the other.
  if (components.scheme != "https" || !components.host.has_value() ||
      !components.path.has_value() || components.query.has_value()) {
    return std::nullopt;
  }

  // An account name has 3 to 24 lowercase letters and digits.
  std::regex static const host_pattern{R"(^([a-z0-9]{3,24})\.blob\.core\.windows\.net$)"};
  std::smatch host_match;
  if (!std::regex_match(components.host.value(), host_match, host_pattern)) { return std::nullopt; }

  // The path must name both a container and a blob.
  std::regex static const path_pattern{R"(^/[^/]+/.+$)"};
  if (!std::regex_match(components.path.value(), path_pattern)) { return std::nullopt; }

  // The signature must use the path exactly as it is sent. Therefore the path is encoded once
  // here, and both the URL and the signature use this encoded path.
  auto encoded_path = detail::UrlEncoder::encode_path(components.path.value());
  components.path   = encoded_path;
  return BlobUrl{.encoded_url  = detail::UrlBuilder::build_manually(components),
                 .account      = host_match[1].str(),
                 .encoded_path = std::move(encoded_path)};
}

}  // namespace

AzureBlobSharedKeyEndpoint::AzureBlobSharedKeyEndpoint(std::string url,
                                                       std::optional<std::string> account_key)
  : RemoteEndpoint{RemoteEndpointType::AZURE_BLOB_SHARED_KEY}, _url{std::move(url)}
{
  KVIKIO_NVTX_FUNC_RANGE();
  auto blob_url = parse_blob_url(_url);
  KVIKIO_EXPECT(blob_url.has_value(),
                "Invalid Azure Blob URL. It must have the form "
                "https://<account>.blob.core.windows.net/<container>/<blob>, with no query.",
                std::invalid_argument);
  _encoded_url  = std::move(blob_url->encoded_url);
  _account      = std::move(blob_url->account);
  _encoded_path = std::move(blob_url->encoded_path);

  if (!account_key.has_value()) {
    auto [env_var_name, env_account_key, is_set] =
      getenv_or({"AZURE_STORAGE_ACCOUNT_KEY", "AZURE_STORAGE_ACCESS_KEY", "AZURE_STORAGE_KEY"},
                std::string{},
                true /* hide_values */);
    KVIKIO_EXPECT(is_set,
                  "Azure Blob Shared Key: must provide `account_key` if none of "
                  "AZURE_STORAGE_ACCOUNT_KEY, AZURE_STORAGE_ACCESS_KEY and AZURE_STORAGE_KEY is "
                  "set.",
                  std::invalid_argument);
    account_key = std::move(env_account_key);
  }
  _account_key = detail::base64_decode(account_key.value());
  KVIKIO_EXPECT(!_account_key.empty(),
                "Azure Blob Shared Key: the account key is empty.",
                std::invalid_argument);
}

std::vector<std::string> AzureBlobSharedKeyEndpoint::request_headers(
  std::string_view verb, std::optional<std::string_view> range) const
{
  auto const date = detail::format_x_ms_date(std::chrono::system_clock::now());
  return detail::azure_shared_key_headers(verb, _account, _account_key, _encoded_path, range, date);
}

void AzureBlobSharedKeyEndpoint::setopt(CurlHandle& curl)
{
  KVIKIO_NVTX_FUNC_RANGE();
  curl.setopt(CURLOPT_URL, _encoded_url.c_str());
}

std::string AzureBlobSharedKeyEndpoint::str() const { return _url; }

std::size_t AzureBlobSharedKeyEndpoint::get_file_size()
{
  KVIKIO_NVTX_FUNC_RANGE();
  auto curl = create_curl_handle();
  setopt(curl);
  curl.setopt(CURLOPT_NOBODY, 1L);
  curl.set_per_attempt_header_builder([this] { return request_headers("HEAD", std::nullopt); });
  {
    detail::ScopedTimer const probe{detail::count_remote_size_probe};
    curl.perform();
  }
  curl_off_t content_length{};
  curl.getinfo(CURLINFO_CONTENT_LENGTH_DOWNLOAD_T, &content_length);
  KVIKIO_EXPECT(content_length >= 0,
                "cannot get size of " + str() + ", content-length not provided by the server",
                std::runtime_error);
  return static_cast<std::size_t>(content_length);
}

void AzureBlobSharedKeyEndpoint::setup_range_request(CurlHandle& curl,
                                                     std::size_t file_offset,
                                                     std::size_t size)
{
  KVIKIO_NVTX_FUNC_RANGE();
  KVIKIO_EXPECT(size > 0, "cannot create a zero-size range request", std::invalid_argument);
  // `request_headers()` puts the range in a signed x-ms-range header. Setting CURLOPT_RANGE as
  // well would add a standard Range header that the signature does not cover, and Azure would
  // reject the request.
  auto range = std::format("bytes={}-{}", file_offset, file_offset + size - 1);
  curl.set_per_attempt_header_builder(
    [this, range = std::move(range)] { return request_headers("GET", range); });
}

bool AzureBlobSharedKeyEndpoint::is_url_valid(std::string const& url) noexcept
{
  try {
    return parse_blob_url(url).has_value();
  } catch (...) {
    return false;
  }
}

AzureBlobOAuthEndpoint::AzureBlobOAuthEndpoint(std::string url, std::optional<std::string> token)
  : RemoteEndpoint{RemoteEndpointType::AZURE_BLOB_OAUTH},
    _url{std::move(url)},
    _headers{nullptr, curl_slist_free_all}
{
  KVIKIO_NVTX_FUNC_RANGE();
  auto blob_url = parse_blob_url(_url);
  KVIKIO_EXPECT(blob_url.has_value(),
                "Invalid Azure Blob URL. It must have the form "
                "https://<account>.blob.core.windows.net/<container>/<blob>, with no query.",
                std::invalid_argument);
  _encoded_url = std::move(blob_url->encoded_url);

  auto const access_token = detail::unwrap_or_env(
    std::move(token),
    "AZURE_STORAGE_TOKEN",
    "Azure Blob OAuth: must provide `token` if AZURE_STORAGE_TOKEN isn't set.");
  KVIKIO_EXPECT(
    !access_token->empty(), "Azure Blob OAuth: the token is empty.", std::invalid_argument);

  // Azure rejects a request with a bearer token unless it also has an `x-ms-version` header.
  for (auto const& header : {std::format("Authorization: Bearer {}", access_token.value()),
                             std::format("x-ms-version: {}", detail::azure_storage_api_version)}) {
    auto* const list_head = curl_slist_append(_headers.get(), header.c_str());
    KVIKIO_EXPECT(list_head != nullptr, "curl_slist_append() failed", std::runtime_error);
    if (_headers == nullptr) { _headers.reset(list_head); }
  }
}

void AzureBlobOAuthEndpoint::setopt(CurlHandle& curl)
{
  KVIKIO_NVTX_FUNC_RANGE();
  curl.setopt(CURLOPT_URL, _encoded_url.c_str());
  curl.setopt(CURLOPT_HTTPHEADER, _headers.get());
}

std::string AzureBlobOAuthEndpoint::str() const { return _url; }

std::size_t AzureBlobOAuthEndpoint::get_file_size()
{
  KVIKIO_NVTX_FUNC_RANGE();
  return detail::get_file_size_using_head_impl(*this, _url);
}

void AzureBlobOAuthEndpoint::setup_range_request(CurlHandle& curl,
                                                 std::size_t file_offset,
                                                 std::size_t size)
{
  KVIKIO_NVTX_FUNC_RANGE();
  detail::setup_range_request_impl(curl, file_offset, size);
}

bool AzureBlobOAuthEndpoint::is_url_valid(std::string const& url) noexcept
{
  try {
    return parse_blob_url(url).has_value();
  } catch (...) {
    return false;
  }
}

}  // namespace kvikio
