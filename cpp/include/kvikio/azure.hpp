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
   * If `account_key` is not given, the key comes from an environment variable. Tools name this
   * variable differently, and KvikIO accepts the names that these tools read:
   * - Azure CLI: `AZURE_STORAGE_KEY`
   * - Python package adlfs: `AZURE_STORAGE_ACCOUNT_KEY`
   * - Rust crate object_store: `AZURE_STORAGE_ACCOUNT_KEY` or `AZURE_STORAGE_ACCESS_KEY`
   *
   * If more than one of these variables is set, all must have the same value.
   *
   * @param url The URL of the blob, `https://<account>.blob.core.windows.net/<container>/<blob>`,
   * without a query.
   * @param account_key The storage account key, in base64 as Azure shows it. If not given, the key
   * comes from one of the environment variables listed above.
   * @exception std::invalid_argument if `url` is not valid for this endpoint, if no account key is
   * given or set in these environment variables, if these environment variables have different
   * values, or if the account key is empty or not valid base64.
   */
  explicit AzureBlobSharedKeyEndpoint(std::string url,
                                      std::optional<std::string> account_key = std::nullopt);

  ~AzureBlobSharedKeyEndpoint() override = default;

  /**
   * @brief Set the URL of the blob on a curl handle.
   *
   * The signed headers are not set here. `get_file_size()` and `setup_range_request()` set them.
   *
   * @param curl The curl handle.
   */
  void setopt(CurlHandle& curl) override;

  /**
   * @brief Get the URL of the blob as given to the constructor.
   *
   * @return The URL.
   */
  std::string str() const override;

  /**
   * @brief Get the size of the blob from a signed HEAD request.
   *
   * @return The size of the blob in bytes.
   */
  std::size_t get_file_size() override;

  /**
   * @brief Set up a curl handle to read part of the blob, with the range in a signed `x-ms-range`
   * header.
   *
   * @param curl The curl handle.
   * @param file_offset The offset in the blob of the first byte to read.
   * @param size The number of bytes to read.
   * @exception std::invalid_argument if `size` is 0.
   */
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

}  // namespace kvikio
