/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <climits>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <string_view>

#include <openssl/evp.h>
#include <openssl/hmac.h>
#include <openssl/sha.h>

#include <kvikio/detail/crypto.hpp>
#include <kvikio/error.hpp>

namespace kvikio::detail {

namespace {

/// OpenSSL takes most buffer sizes as `int`.
int checked_int_size(std::size_t size)
{
  KVIKIO_EXPECT(size <= static_cast<std::size_t>(INT_MAX),
                "Input is too large for OpenSSL.",
                std::invalid_argument);
  return static_cast<int>(size);
}

unsigned char const* as_const_uchar(std::string_view s)
{
  return reinterpret_cast<unsigned char const*>(s.data());
}

unsigned char* as_uchar(std::string& s) { return reinterpret_cast<unsigned char*>(s.data()); }

}  // namespace

std::string hmac_sha256(std::string_view key, std::string_view message)
{
  std::string digest(SHA256_DIGEST_LENGTH, '\0');
  auto const* result = HMAC(EVP_sha256(),
                            key.data(),
                            checked_int_size(key.size()),
                            as_const_uchar(message),
                            message.size(),
                            as_uchar(digest),
                            nullptr);
  KVIKIO_EXPECT(result != nullptr, "HMAC-SHA256 failed.", std::runtime_error);
  return digest;
}

std::string base64_encode(std::string_view data)
{
  // Every 3 input bytes, with the last group rounded up, become 4 characters. EVP_EncodeBlock()
  // also writes a null terminator after them.
  std::string text(4 * ((data.size() + 2) / 3) + 1, '\0');
  // EVP_EncodeBlock() returns the text size as an `int`. The text is longer than the data.
  // Therefore this check also covers the data size.
  checked_int_size(text.size());
  auto const text_size =
    EVP_EncodeBlock(as_uchar(text), as_const_uchar(data), static_cast<int>(data.size()));
  text.resize(text_size);
  return text;
}

std::string base64_decode(std::string_view text)
{
  std::size_t num_padding{0};
  if (text.ends_with("==")) {
    num_padding = 2;
  } else if (text.ends_with('=')) {
    num_padding = 1;
  }

  // EVP_DecodeBlock() silently trims whitespace at both ends and decodes `=` even in the middle.
  // This check rejects both cases. Therefore `num_padding` is exactly the number of extra bytes to
  // cut off below.
  constexpr std::string_view alphabet{
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"};
  auto const unpadded       = text.substr(0, text.size() - num_padding);
  bool const is_in_alphabet = unpadded.find_first_not_of(alphabet) == std::string_view::npos;
  KVIKIO_EXPECT(text.size() % 4 == 0 && is_in_alphabet,
                "Invalid base64 text. It must consist of A-Z, a-z, 0-9, '+' and '/', followed by "
                "at most two '=', and its length must be a multiple of 4.",
                std::invalid_argument);

  // EVP_DecodeBlock() writes 3 bytes for every 4 characters, including a zero byte for each `=`.
  // The resize below cuts off those zero bytes.
  std::string data(text.size() / 4 * 3, '\0');
  auto const decoded_size =
    EVP_DecodeBlock(as_uchar(data), as_const_uchar(text), checked_int_size(text.size()));
  KVIKIO_EXPECT(decoded_size >= 0, "EVP_DecodeBlock() failed.", std::runtime_error);
  data.resize(static_cast<std::size_t>(decoded_size) - num_padding);
  return data;
}

}  // namespace kvikio::detail
