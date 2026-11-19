/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cstddef>
#include <format>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include <kvikio/detail/crypto.hpp>

using kvikio::detail::base64_decode;
using kvikio::detail::base64_encode;
using kvikio::detail::hmac_sha256;

namespace {

std::vector<std::byte> to_bytes(std::string_view text)
{
  auto const bytes = std::as_bytes(std::span{text});
  return {bytes.begin(), bytes.end()};
}

std::string to_hex(std::vector<std::byte> const& bytes)
{
  std::string hex;
  for (auto const byte : bytes) {
    hex += std::format("{:02x}", std::to_integer<unsigned int>(byte));
  }
  return hex;
}

}  // namespace

TEST(CryptoTest, base64_matches_rfc_4648)
{
  // Test vectors from RFC 4648 (https://www.rfc-editor.org/info/rfc4648/), section 10. The inputs
  // of 1, 2 and 3 bytes end with two, one and zero `=`.
  std::vector<std::pair<std::string_view, std::string_view>> const vectors{
    {"", ""}, {"f", "Zg=="}, {"fo", "Zm8="}, {"foo", "Zm9v"}};
  for (auto const& [data, text] : vectors) {
    EXPECT_EQ(base64_encode(to_bytes(data)), text);
    EXPECT_EQ(base64_decode(text), to_bytes(data)) << text;
  }
}

TEST(CryptoTest, base64_decode_rejects_invalid_text)
{
  for (std::string_view const text : {
         "Zm9vYmF",    // The length is not a multiple of 4.
         "Zg==Zm9v",   // `=` is in the middle.
         " Zm9vYmF",   // Whitespace leads.
         "Zm9v\nYmF",  // Whitespace is embedded.
         "Zm9vYmF-",   // `-` is an invalid character.
         "Zh==",       // This text decodes to "f", but an encoder writes "f" as `Zg==`.
       }) {
    EXPECT_THROW(std::ignore = base64_decode(text), std::invalid_argument) << text;
  }
}

TEST(CryptoTest, hmac_sha256_matches_rfc_4231)
{
  // Test case 2 from RFC 4231 (https://www.rfc-editor.org/info/rfc4231/), section 4.3.
  EXPECT_EQ(to_hex(hmac_sha256(to_bytes("Jefe"), "what do ya want for nothing?")),
            "5bdcc146bf60754e6a042426089575c75a003f089d2739839dec58b964ec3843");
}
