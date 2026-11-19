/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <chrono>
#include <cstddef>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace kvikio::detail {

/// The Azure Storage REST API version that KvikIO sends in the `x-ms-version` header. See
/// <https://learn.microsoft.com/en-us/rest/api/storageservices/versioning-for-the-azure-storage-services>.
constexpr std::string_view azure_storage_api_version{"2026-04-06"};

/**
 * @brief Format a time as the `x-ms-date` header expects, for example
 * `"Tue, 17 Nov 2026 03:04:05 GMT"`.
 *
 * The day and month names are always English, whatever the locale of the process is.
 *
 * @param time The time to format.
 * @return The formatted time in GMT.
 */
[[nodiscard]] std::string format_x_ms_date(std::chrono::system_clock::time_point time);

/**
 * @brief Build the headers of one Azure Blob Storage request authorized with Shared Key.
 *
 * The returned headers are `x-ms-date`, `x-ms-range` (if `range` is given), `x-ms-version` and
 * `Authorization`. The `Authorization` header holds a signature over the other three and over
 * `path`.
 *
 * See <https://learn.microsoft.com/en-us/rest/api/storageservices/authorize-with-shared-key>.
 *
 * @param verb The HTTP method, such as `"GET"` or `"HEAD"`.
 * @param account The storage account name.
 * @param key The account key, already decoded from base64.
 * @param path The URL path of the blob, `"/<container>/<blob>"`, encoded exactly as in the request
 * URL.
 * @param range The value of the `x-ms-range` header, such as `"bytes=0-15"`, or `std::nullopt` to
 * send no range.
 * @param date The value of the `x-ms-date` header, as returned by `format_x_ms_date()`.
 * @return The header lines, each formatted as `"<name>: <value>"`.
 */
[[nodiscard]] std::vector<std::string> azure_shared_key_headers(
  std::string_view verb,
  std::string_view account,
  std::span<std::byte const> key,
  std::string_view path,
  std::optional<std::string_view> range,
  std::string_view date);

}  // namespace kvikio::detail
