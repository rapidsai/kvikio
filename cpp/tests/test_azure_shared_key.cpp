/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <chrono>
#include <cstddef>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <kvikio/detail/azure_shared_key.hpp>
#include <kvikio/detail/crypto.hpp>

using kvikio::detail::azure_shared_key_headers;
using kvikio::detail::base64_decode;
using kvikio::detail::format_x_ms_date;

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
  // A made-up key that decodes to 64 bytes, like a real account key. These shell commands
  // computed the expected signature:
  //   KEY=<as below>
  //   HEXKEY=$(printf '%s' "$KEY" | base64 -d | od -An -v -tx1 | tr -d ' \n')
  //   STRING_TO_SIGN=$(printf '%s\n' GET "" "" "" "" "" "" "" "" "" "" ""
  //     "x-ms-date:Tue, 17 Nov 2026 03:04:05 GMT" "x-ms-range:bytes=0-15"
  //     "x-ms-version:2026-04-06" "/kvikiodevstandard/basic/32MiB.bin")
  //   printf '%s' "$STRING_TO_SIGN" |
  //     openssl dgst -sha256 -mac HMAC -macopt "hexkey:$HEXKEY" -binary | base64 -w0
  std::vector<std::byte> const key = base64_decode(
    "Wraiths+Nekkers+Drowners+Griffins+Leshens+Fiends+Botchlings+Bruxae+Ghouls+Hym+GraveHag==");
  std::string const expected_signature = "B0Kd0gYizxIzDQCVcgsZB69Lk/voAPzvqIisX/ZNEGo=";

  auto const headers = azure_shared_key_headers("GET",
                                                "kvikiodevstandard",
                                                key,
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
