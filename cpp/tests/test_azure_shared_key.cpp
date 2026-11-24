/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <chrono>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include <gtest/gtest.h>

#include <kvikio/detail/azure_shared_key.hpp>
#include <kvikio/detail/crypto.hpp>

using kvikio::detail::azure_shared_key_headers;
using kvikio::detail::base64_decode;
using kvikio::detail::format_x_ms_date;

namespace {

// A made-up account key that decodes to 64 bytes, like a real one. These shell commands computed
// the expected signatures below, with LINES set to the lines of the string to sign that each test
// gives:
//   KEY=<as below>
//   HEXKEY=$(printf '%s' "$KEY" | base64 -d | od -An -v -tx1 | tr -d ' \n')
//   LINES=(<as given in the test>)
//   STRING_TO_SIGN=$(printf '%s\n' "${LINES[@]}")
//   printf '%s' "$STRING_TO_SIGN" |
//     openssl dgst -sha256 -mac HMAC -macopt "hexkey:$HEXKEY" -binary | base64 -w0
constexpr std::string_view test_account_key{
  "Wraiths+Nekkers+Drowners+Griffins+Leshens+Fiends+Botchlings+Bruxae+Ghouls+Hym+GraveHag=="};

}  // namespace

TEST(AzureSharedKeyTest, x_ms_date_is_english_gmt)
{
  using namespace std::chrono;
  auto const time =
    sys_days{year{2026} / 11 / 17} + hours{3} + minutes{4} + seconds{5} + milliseconds{678};
  EXPECT_EQ(format_x_ms_date(time), "Tue, 17 Nov 2026 03:04:05 GMT")
    << "the fraction of a second is dropped";
}

TEST(AzureSharedKeyTest, headers_are_correct)
{
  //   LINES=(GET "" "" "" "" "" "" "" "" "" "" ""
  //     "x-ms-date:Tue, 17 Nov 2026 03:04:05 GMT" "x-ms-range:bytes=0-15"
  //     "x-ms-version:2026-04-06" "/kvikiodevstandard/basic/32MiB.bin")
  std::string const expected_signature = "B0Kd0gYizxIzDQCVcgsZB69Lk/voAPzvqIisX/ZNEGo=";

  auto const headers = azure_shared_key_headers("GET",
                                                "kvikiodevstandard",
                                                base64_decode(test_account_key),
                                                "/basic/32MiB.bin",
                                                "bytes=0-15",
                                                "Tue, 17 Nov 2026 03:04:05 GMT");

  std::vector<std::string> const expected{
    "x-ms-date: Tue, 17 Nov 2026 03:04:05 GMT",
    "x-ms-range: bytes=0-15",
    "x-ms-version: 2026-04-06",
    "Authorization: SharedKey kvikiodevstandard:" + expected_signature};
  EXPECT_EQ(headers, expected);
}

TEST(AzureSharedKeyTest, headers_without_range_are_correct)
{
  //   LINES=(HEAD "" "" "" "" "" "" "" "" "" "" ""
  //     "x-ms-date:Tue, 17 Nov 2026 03:04:05 GMT"
  //     "x-ms-version:2026-04-06" "/kvikiodevstandard/basic/32MiB.bin")
  std::string const expected_signature = "t9dIzMSIEsIi3wayBzE9DOWO1TolymwQ/aFcDBhb9S4=";

  auto const headers = azure_shared_key_headers("HEAD",
                                                "kvikiodevstandard",
                                                base64_decode(test_account_key),
                                                "/basic/32MiB.bin",
                                                std::nullopt,
                                                "Tue, 17 Nov 2026 03:04:05 GMT");

  std::vector<std::string> const expected{
    "x-ms-date: Tue, 17 Nov 2026 03:04:05 GMT",
    "x-ms-version: 2026-04-06",
    "Authorization: SharedKey kvikiodevstandard:" + expected_signature};
  EXPECT_EQ(headers, expected);
}
