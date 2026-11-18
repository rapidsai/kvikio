/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace kvikio::detail {

/**
 * @brief Compute the HMAC-SHA256 of a message.
 *
 * @param key The secret key.
 * @param message The message to authenticate.
 * @return The 32-byte HMAC.
 * @exception std::invalid_argument if `key` has more bytes than an `int` can hold.
 * @exception std::runtime_error if OpenSSL fails to compute the HMAC.
 */
[[nodiscard]] std::vector<std::byte> hmac_sha256(std::span<std::byte const> key,
                                                 std::string_view message);

/**
 * @brief Encode bytes as base64 text, with `=` padding and without line breaks.
 *
 * @param data The bytes to encode.
 * @return The base64 text.
 * @exception std::invalid_argument if the base64 text is too long for OpenSSL, which counts it as
 * an `int`.
 */
[[nodiscard]] std::string base64_encode(std::span<std::byte const> data);

/**
 * @brief Decode base64 text into bytes.
 *
 * @param text Base64 text made of `A-Z`, `a-z`, `0-9`, `+` and `/`, followed by at most two `=`.
 * Its length must be a multiple of 4. Whitespace is not allowed anywhere.
 * @return The decoded bytes.
 * @exception std::invalid_argument if `text` is not valid base64 or has more characters than an
 * `int` can hold.
 * @exception std::runtime_error if OpenSSL fails to decode `text`.
 */
[[nodiscard]] std::vector<std::byte> base64_decode(std::string_view text);

}  // namespace kvikio::detail
