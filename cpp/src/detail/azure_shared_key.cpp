/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <chrono>
#include <format>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <kvikio/detail/azure_shared_key.hpp>
#include <kvikio/detail/crypto.hpp>

namespace kvikio::detail {

std::string format_x_ms_date(std::chrono::system_clock::time_point time)
{
  // Without the `L` option, std::format() uses the "C" locale. Therefore the day and month names
  // are always English. Rounding down to whole seconds keeps %S from printing a fraction.
  return std::format("{:%a, %d %b %Y %H:%M:%S} GMT",
                     std::chrono::floor<std::chrono::seconds>(time));
}

std::vector<std::string> azure_shared_key_headers(std::string_view verb,
                                                  std::string_view account,
                                                  std::string_view key,
                                                  std::string_view path,
                                                  std::optional<std::string_view> range,
                                                  std::string_view date)
{
  // The string to sign lists the x-ms-* headers sorted by name.
  std::vector<std::pair<std::string_view, std::string_view>> ms_headers;
  ms_headers.emplace_back("x-ms-date", date);
  if (range.has_value()) { ms_headers.emplace_back("x-ms-range", range.value()); }
  ms_headers.emplace_back("x-ms-version", azure_storage_api_version);

  std::string string_to_sign{verb};
  string_to_sign += '\n';
  // One empty line for each of the 11 standard headers from Content-Encoding to Range. KvikIO sends
  // none of them.
  string_to_sign.append(11, '\n');
  for (auto const& [name, value] : ms_headers) {
    string_to_sign.append(name).append(":").append(value).append("\n");
  }
  string_to_sign.append("/").append(account).append(path);

  auto const signature = base64_encode(hmac_sha256(key, string_to_sign));

  std::vector<std::string> headers;
  for (auto const& [name, value] : ms_headers) {
    headers.push_back(std::string{name} + ": " + std::string{value});
  }
  headers.push_back("Authorization: SharedKey " + std::string{account} + ":" + signature);
  return headers;
}

}  // namespace kvikio::detail
