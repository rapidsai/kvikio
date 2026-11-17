/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <chrono>
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

TEST(AzureSharedKeyTest, headers_match_reference_script)
{
  // A made-up key: the 64 bytes 0x00 to 0x3f. The expected signature comes from a shell script that
  // signs with openssl and has read a real blob from Azure. Only its key and date were replaced.
  std::string const key = base64_decode(
    "AAECAwQFBgcICQoLDA0ODxAREhMUFRYXGBkaGxwdHh8gISIjJCUmJygpKissLS4vMDEy"
    "MzQ1Njc4OTo7PD0+Pw==");
  std::string const expected_signature = "D1Nwl4x88zi71ijVDoN7EonApOXAkOjRHMqhnQgutoY=";

  auto const headers = azure_shared_key_headers("GET",
                                                "kvikiodevstandard",
                                                key,
                                                "/basic/32MiB.bin",
                                                "bytes=0-15",
                                                "Tue, 17 Nov 2026 03:04:05 GMT");

  std::vector<std::string> const expected{
    "x-ms-date: Tue, 17 Nov 2026 03:04:05 GMT",
    "x-ms-range: bytes=0-15",
    "x-ms-version: 2021-08-06",
    "Authorization: SharedKey kvikiodevstandard:" + expected_signature};
  EXPECT_EQ(headers, expected);
}
