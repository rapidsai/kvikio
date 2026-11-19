/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cstddef>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <openssl/evp.h>
#include <openssl/hmac.h>
#include <openssl/sha.h>

#include <kvikio/detail/crypto.hpp>
#include <kvikio/error.hpp>

namespace kvikio::detail {

namespace {

/// OpenSSL takes most buffer sizes as `int`.
int check_int_size(std::size_t size)
{
  KVIKIO_EXPECT(size <= static_cast<std::size_t>(std::numeric_limits<int>::max()),
                "Input is too large for OpenSSL.",
                std::invalid_argument);
  return static_cast<int>(size);
}

unsigned char const* const_uchar_data(std::string_view s)
{
  return reinterpret_cast<unsigned char const*>(s.data());
}

unsigned char const* const_uchar_data(std::span<std::byte const> bytes)
{
  return reinterpret_cast<unsigned char const*>(bytes.data());
}

unsigned char* uchar_data(std::string& s) { return reinterpret_cast<unsigned char*>(s.data()); }

unsigned char* uchar_data(std::vector<std::byte>& bytes)
{
  return reinterpret_cast<unsigned char*>(bytes.data());
}

}  // namespace

std::vector<std::byte> hmac_sha256(std::span<std::byte const> key, std::string_view message)
{
  std::vector<std::byte> digest(SHA256_DIGEST_LENGTH);
  auto const* result = HMAC(EVP_sha256(),
                            key.data(),
                            check_int_size(key.size()),
                            const_uchar_data(message),
                            message.size(),
                            uchar_data(digest),
                            nullptr);
  KVIKIO_EXPECT(result != nullptr, "HMAC-SHA256 failed.", std::runtime_error);
  return digest;
}

std::string base64_encode(std::span<std::byte const> data)
{
  // Every 3 input bytes, with the last group rounded up, become 4 characters. EVP_EncodeBlock()
  // also writes a null terminator after them.
  std::string text(4 * ((data.size() + 2) / 3) + 1, '\0');
  // EVP_EncodeBlock() returns the text size as an `int`. The text is longer than the data.
  // Therefore this overflow check also covers the data size.
  check_int_size(text.size());
  auto const text_size =
    EVP_EncodeBlock(uchar_data(text), const_uchar_data(data), static_cast<int>(data.size()));
  text.resize(text_size);
  return text;
}

std::vector<std::byte> base64_decode(std::string_view text)
{
  std::size_t num_padding{0};
  if (text.ends_with("==")) {
    num_padding = 2;
  } else if (text.ends_with('=')) {
    num_padding = 1;
  }

  // EVP_DecodeBlock() writes 3 bytes for every 4 characters, including a zero byte for each `=`
  // (padding during encoding). The resize below cuts off those zero bytes.
  std::vector<std::byte> data(text.size() / 4 * 3);
  auto const decoded_size =
    EVP_DecodeBlock(uchar_data(data), const_uchar_data(text), check_int_size(text.size()));
  KVIKIO_EXPECT(decoded_size >= 0, "Invalid base64 text.", std::invalid_argument);
  data.resize(static_cast<std::size_t>(decoded_size) - num_padding);

  // EVP_DecodeBlock() also accepts some invalid text, such as `Zh==`. Only valid text comes back
  // unchanged when the decoded bytes are encoded again.
  KVIKIO_EXPECT(base64_encode(data) == text, "Invalid base64 text.", std::invalid_argument);
  return data;
}

}  // namespace kvikio::detail
