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
  // Azure expects an RFC 1123 date, such as "Tue, 17 Nov 2026 03:04:05 GMT".
  // - %a        abbreviated weekday name, "Tue"
  // - %d        two-digit day of the month, "17"
  // - %b        abbreviated month name, "Nov"
  // - %Y        four-digit year, "2026"
  // - %H:%M:%S  24-hour time, "03:04:05"
  // The format string writes "GMT" literally because %Z would print "UTC" for a system_clock time.
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
  std::string string_to_sign = std::format("{}\n", verb);

  // The string to sign has one line for each of these 11 standard headers, in this order:
  //   Content-Encoding, Content-Language, Content-Length, Content-MD5, Content-Type, Date,
  //   If-Modified-Since, If-Match, If-None-Match, If-Unmodified-Since, Range
  // KvikIO sends none of them. Therefore all 11 lines are empty. In particular:
  // - The Date line is empty due to the use of x-ms-date.
  // - The Range line is empty due to the use of x-ms-range, which is signed with the other x-ms-*
  //   headers.
  string_to_sign.append(11, '\n');

  // The string to sign lists the x-ms-* headers sorted by name.
  std::vector<std::pair<std::string_view, std::string_view>> ms_headers;
  ms_headers.emplace_back("x-ms-date", date);
  if (range.has_value()) { ms_headers.emplace_back("x-ms-range", range.value()); }
  ms_headers.emplace_back("x-ms-version", azure_storage_api_version);
  for (auto const& [name, value] : ms_headers) {
    string_to_sign += std::format("{}:{}\n", name, value);
  }
  string_to_sign += std::format("/{}{}", account, path);

  auto const signature = base64_encode(hmac_sha256(key, string_to_sign));

  std::vector<std::string> headers;
  for (auto const& [name, value] : ms_headers) {
    headers.push_back(std::format("{}: {}", name, value));
  }
  headers.push_back(std::format("Authorization: SharedKey {}:{}", account, signature));
  return headers;
}

}  // namespace kvikio::detail
