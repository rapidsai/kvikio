/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include <kvikio/remote_handle.hpp>

namespace kvikio {

/**
 * @brief A remote endpoint for Azure Blob Storage that signs every request with the storage account
 * key. Azure calls this way of signing "Shared Key".
 */
class AzureBlobSharedKeyEndpoint : public RemoteEndpoint {
 private:
  // The URL as given by the caller.
  std::string _url;
  // The URL that is sent, with special characters in the path encoded.
  std::string _encoded_url;
  // The storage account name.
  std::string _account;
  // The path of `_encoded_url`, "/<container>/<blob>".
  std::string _encoded_path;
  // The account key, decoded from base64.
  std::vector<std::byte> _account_key;

  /**
   * @brief Build the headers for one request.
   *
   * KvikIO calls this function right before it sends each request, including each retry.
   * Therefore a signature based on the current time is always fresh when the request goes out.
   *
   * @param verb The HTTP method, `"GET"` or `"HEAD"`.
   * @param range The value of the `x-ms-range` header, such as `"bytes=0-15"`, or `std::nullopt` to
   * send no range.
   * @return The header lines, each formatted as `"<name>: <value>"`.
   */
  [[nodiscard]] std::vector<std::string> request_headers(
    std::string_view verb, std::optional<std::string_view> range) const;

 public:
  /**
   * @brief Create an Azure Blob endpoint that signs every request with the storage account key.
   *
   * @param url The URL of the blob, `https://<account>.blob.core.windows.net/<container>/<blob>`,
   * without a query.
   * @param account_key The storage account key, in base64 as Azure shows it. If not given, the
   * value of the `AZURE_STORAGE_ACCOUNT_KEY` environment variable is used.
   * @exception std::invalid_argument if `url` is not valid for this endpoint, if no account key is
   * given or set in `AZURE_STORAGE_ACCOUNT_KEY`, or if the account key is empty or not valid
   * base64.
   */
  explicit AzureBlobSharedKeyEndpoint(std::string url,
                                      std::optional<std::string> account_key = std::nullopt);

  ~AzureBlobSharedKeyEndpoint() override = default;
  void setopt(CurlHandle& curl) override;
  std::string str() const override;
  std::size_t get_file_size() override;
  void setup_range_request(CurlHandle& curl, std::size_t file_offset, std::size_t size) override;

  /**
   * @brief Whether the given URL is valid for the Azure Blob endpoint with Shared Key.
   *
   * A valid URL has the form `https://<account>.blob.core.windows.net/<container>/<blob>` and has
   * no query.
   *
   * @param url A URL.
   * @return Boolean answer.
   */
  static bool is_url_valid(std::string const& url) noexcept;
};

/**
 * @brief A remote endpoint for Azure Blob Storage that sends a Microsoft Entra ID access token with
 * every request.
 *
 * The token is an OAuth 2.0 bearer token. KvikIO sends it as given and never renews it. Therefore
 * the token must stay valid until the last read through this endpoint ends.
 */
class AzureBlobOAuthEndpoint : public RemoteEndpoint {
 private:
  // The URL as given by the caller.
  std::string _url;
  // The URL that is sent, with special characters in the path encoded.
  std::string _encoded_url;
  // The `Authorization` and `x-ms-version` headers, as the list that libcurl takes.
  // `curl_slist_free_all()` frees the list.
  std::unique_ptr<curl_slist, void (*)(curl_slist*)> _headers;

 public:
  /**
   * @brief Create an Azure Blob endpoint that sends a Microsoft Entra ID access token with every
   * request.
   *
   * @param url The URL of the blob, `https://<account>.blob.core.windows.net/<container>/<blob>`,
   * without a query.
   * @param token The access token, issued for Azure Storage (`https://storage.azure.com/`). If not
   * given, the value of the `AZURE_STORAGE_TOKEN` environment variable is used.
   * @exception std::invalid_argument if `url` is not valid for this endpoint, if no token is given
   * or set in `AZURE_STORAGE_TOKEN`, or if the token is empty.
   */
  explicit AzureBlobOAuthEndpoint(std::string url, std::optional<std::string> token = std::nullopt);

  ~AzureBlobOAuthEndpoint() override = default;
  void setopt(CurlHandle& curl) override;
  std::string str() const override;
  std::size_t get_file_size() override;
  void setup_range_request(CurlHandle& curl, std::size_t file_offset, std::size_t size) override;

  /**
   * @brief Whether the given URL is valid for the Azure Blob endpoint with OAuth.
   *
   * A valid URL has the form `https://<account>.blob.core.windows.net/<container>/<blob>` and has
   * no query.
   *
   * @param url A URL.
   * @return Boolean answer.
   */
  static bool is_url_valid(std::string const& url) noexcept;
};

}  // namespace kvikio
