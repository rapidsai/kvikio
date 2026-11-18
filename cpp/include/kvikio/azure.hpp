/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cstddef>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include <kvikio/remote_handle.hpp>

namespace kvikio {

/**
 * @brief Base class of the remote endpoints for Azure Blob Storage.
 *
 * The URL of a blob has the form `https://<account>.blob.core.windows.net/<container>/<blob>`. Each
 * subclass authorizes requests in its own way, through the headers that `request_headers()`
 * returns.
 */
class AzureBlobEndpoint : public RemoteEndpoint {
 private:
  std::string _url;           // The URL as given by the caller.
  std::string _encoded_url;   // The URL that is sent, with special characters in the path encoded.
  std::string _account;       // The storage account name.
  std::string _encoded_path;  // The path of `_encoded_url`, "/<container>/<blob>".

 protected:
  /**
   * @brief Create an Azure Blob endpoint from a URL.
   *
   * @param remote_endpoint_type The type of the subclass.
   * @param url The URL of the blob.
   * @exception std::invalid_argument if `url` is not a valid Azure Blob URL.
   */
  AzureBlobEndpoint(RemoteEndpointType remote_endpoint_type, std::string url);

  /// The storage account name. For a URL on the secondary host `<account>-secondary`, this is the
  /// name of the primary account.
  [[nodiscard]] std::string const& account() const noexcept;

  /// The URL path that is sent, "/<container>/<blob>", with special characters encoded.
  [[nodiscard]] std::string const& encoded_path() const noexcept;

  /**
   * @brief Build the headers of one request attempt.
   *
   * This is called right before each attempt, including retries. Therefore it can include headers
   * that expire, such as a signature over the current time.
   *
   * @param verb The HTTP method, `"GET"` or `"HEAD"`.
   * @param range The value of the `x-ms-range` header, such as `"bytes=0-15"`, or `std::nullopt` to
   * send no range.
   * @return The header lines, each formatted as `"<name>: <value>"`.
   */
  [[nodiscard]] virtual std::vector<std::string> request_headers(
    std::string_view verb, std::optional<std::string_view> range) const = 0;

 public:
  ~AzureBlobEndpoint() override = default;
  void setopt(CurlHandle& curl) override;
  std::string str() const override;
  std::size_t get_file_size() override;
  void setup_range_request(CurlHandle& curl, std::size_t file_offset, std::size_t size) override;

  /**
   * @brief Whether the given URL is valid for Azure Blob endpoints.
   *
   * A valid URL has the form `https://<account>.blob.core.windows.net/<container>/<blob>`,
   * optionally followed by a query. Each subclass can be stricter.
   *
   * @param url A URL.
   * @return Boolean answer.
   */
  static bool is_url_valid(std::string const& url) noexcept;
};

/**
 * @brief A remote endpoint for Azure Blob Storage, authorized with Shared Key.
 *
 * Every request is signed with the storage account key.
 */
class AzureBlobSharedKeyEndpoint : public AzureBlobEndpoint {
 private:
  std::vector<std::byte> _key;  // The account key, decoded from base64.

 protected:
  [[nodiscard]] std::vector<std::string> request_headers(
    std::string_view verb, std::optional<std::string_view> range) const override;

 public:
  /**
   * @brief Create an Azure Blob endpoint authorized with Shared Key.
   *
   * @param url The URL of the blob, `https://<account>.blob.core.windows.net/<container>/<blob>`,
   * without a query.
   * @param account_key The storage account key, in base64 as Azure shows it. If nullopt, the value
   * of the `AZURE_STORAGE_ACCOUNT_KEY` environment variable is used.
   * @exception std::invalid_argument if `url` is not valid for this endpoint, if no account key is
   * given or set in `AZURE_STORAGE_ACCOUNT_KEY`, or if the account key is empty or not valid
   * base64.
   */
  explicit AzureBlobSharedKeyEndpoint(std::string url,
                                      std::optional<std::string> account_key = std::nullopt);

  ~AzureBlobSharedKeyEndpoint() override = default;

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

}  // namespace kvikio
