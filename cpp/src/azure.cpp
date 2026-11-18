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
#include <kvikio/detail/azure_shared_key.hpp>
#include <kvikio/detail/crypto.hpp>
#include <kvikio/detail/env.hpp>
#include <kvikio/detail/nvtx.hpp>
#include <kvikio/detail/url.hpp>
#include <kvikio/error.hpp>
#include <kvikio/remote_handle.hpp>
#include <kvikio/shim/libcurl.hpp>
#include <kvikio/statistics/counters.hpp>

namespace kvikio {

namespace {

/// The parts of an Azure Blob URL that the endpoints need.
struct BlobUrl {
  std::string encoded_url;
  std::string account;
  std::string encoded_path;
  bool has_query;
};

/**
 * @brief Parse an Azure Blob URL.
 *
 * @param url A URL.
 * @return The parts of `url`, or std::nullopt if `url` does not have the form
 * `https://<account>.blob.core.windows.net/<container>/<blob>`, optionally followed by a query.
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
  if (components.scheme != "https" || !components.host.has_value() ||
      !components.path.has_value()) {
    return std::nullopt;
  }

  // An account name has 3 to 24 lowercase letters and digits. The secondary host of an account with
  // read-access geo-redundant storage appends "-secondary" to the name. Requests to it are still
  // authorized with the name of the primary account.
  std::regex static const host_pattern{
    R"(^([a-z0-9]{3,24})(-secondary)?\.blob\.core\.windows\.net$)"};
  std::smatch host_match;
  if (!std::regex_match(components.host.value(), host_match, host_pattern)) { return std::nullopt; }

  // The path must name both a container and a blob.
  std::regex static const path_pattern{R"(^/[^/]+/.+$)"};
  if (!std::regex_match(components.path.value(), path_pattern)) { return std::nullopt; }

  // A request is signed over the path exactly as it is sent. Therefore the path is encoded once
  // here, and the same encoded path is used both in the URL and in the signature.
  components.path = detail::UrlEncoder::encode_path(components.path.value());
  return BlobUrl{.encoded_url  = detail::UrlBuilder::build_manually(components),
                 .account      = host_match[1].str(),
                 .encoded_path = components.path.value(),
                 .has_query    = components.query.has_value()};
}

}  // namespace

AzureBlobEndpoint::AzureBlobEndpoint(RemoteEndpointType remote_endpoint_type, std::string url)
  : RemoteEndpoint{remote_endpoint_type}, _url{std::move(url)}
{
  auto blob_url = parse_blob_url(_url);
  KVIKIO_EXPECT(blob_url.has_value(),
                "Invalid Azure Blob URL. It must have the form "
                "https://<account>.blob.core.windows.net/<container>/<blob>.",
                std::invalid_argument);
  _encoded_url  = std::move(blob_url->encoded_url);
  _account      = std::move(blob_url->account);
  _encoded_path = std::move(blob_url->encoded_path);
}

std::string const& AzureBlobEndpoint::account() const noexcept { return _account; }

std::string const& AzureBlobEndpoint::encoded_path() const noexcept { return _encoded_path; }

void AzureBlobEndpoint::setopt(CurlHandle& curl)
{
  KVIKIO_NVTX_FUNC_RANGE();
  curl.setopt(CURLOPT_URL, _encoded_url.c_str());
}

std::string AzureBlobEndpoint::str() const { return _url; }

std::size_t AzureBlobEndpoint::get_file_size()
{
  KVIKIO_NVTX_FUNC_RANGE();
  // The request is signed over its verb. Therefore this function sends its own HEAD request instead
  // of using `get_file_size_using_head_impl()`, which never tells the endpoint that the request is
  // a HEAD.
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

void AzureBlobEndpoint::setup_range_request(CurlHandle& curl,
                                            std::size_t file_offset,
                                            std::size_t size)
{
  KVIKIO_NVTX_FUNC_RANGE();
  KVIKIO_EXPECT(size > 0, "cannot create a zero-size range request", std::invalid_argument);
  // The range is sent as x-ms-range, among the headers that `request_headers()` builds. A Shared
  // Key signature covers every header that is sent. Therefore this function must not also set
  // CURLOPT_RANGE, which sends a standard Range header.
  auto range = std::format("bytes={}-{}", file_offset, file_offset + size - 1);
  curl.set_per_attempt_header_builder(
    [this, range = std::move(range)] { return request_headers("GET", range); });
}

bool AzureBlobEndpoint::is_url_valid(std::string const& url) noexcept
{
  try {
    return parse_blob_url(url).has_value();
  } catch (...) {
    return false;
  }
}

AzureBlobSharedKeyEndpoint::AzureBlobSharedKeyEndpoint(std::string url,
                                                       std::optional<std::string> account_key)
  : AzureBlobEndpoint{RemoteEndpointType::AZURE_BLOB_SHARED_KEY, std::move(url)}
{
  KVIKIO_NVTX_FUNC_RANGE();
  KVIKIO_EXPECT(is_url_valid(str()),
                "Azure Blob Shared Key: the URL must not have a query.",
                std::invalid_argument);
  auto const key = detail::unwrap_or_env(
    std::move(account_key),
    "AZURE_STORAGE_ACCOUNT_KEY",
    "Azure Blob Shared Key: must provide `account_key` if AZURE_STORAGE_ACCOUNT_KEY isn't set.");
  _key = detail::base64_decode(key.value());
  KVIKIO_EXPECT(
    !_key.empty(), "Azure Blob Shared Key: the account key is empty.", std::invalid_argument);
}

bool AzureBlobSharedKeyEndpoint::is_url_valid(std::string const& url) noexcept
{
  try {
    // A Shared Key signature would also have to cover the query, which
    // `detail::azure_shared_key_headers()` does not support.
    auto const blob_url = parse_blob_url(url);
    return blob_url.has_value() && !blob_url->has_query;
  } catch (...) {
    return false;
  }
}

std::vector<std::string> AzureBlobSharedKeyEndpoint::request_headers(
  std::string_view verb, std::optional<std::string_view> range) const
{
  auto const date = detail::format_x_ms_date(std::chrono::system_clock::now());
  return detail::azure_shared_key_headers(verb, account(), _key, encoded_path(), range, date);
}

}  // namespace kvikio
