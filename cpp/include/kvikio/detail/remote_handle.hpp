/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cstring>

#include <kvikio/shim/libcurl.hpp>

namespace kvikio::detail {
/**
 * @brief Callback for `CURLOPT_WRITEFUNCTION` that copies received data into a `std::string`.
 *
 * @param data Received data
 * @param size Curl internal implementation always sets this parameter to 1
 * @param num_bytes Number of bytes received
 * @param userdata Must be cast from `std::string*`
 * @return The number of bytes consumed by the callback
 */
std::size_t callback_get_string_response(char* data,
                                         std::size_t size,
                                         std::size_t num_bytes,
                                         void* userdata);

/**
 * @brief Set up the range request for libcurl. Use this method when HTTP range request is supposed.
 *
 * @param curl A curl handle
 * @param file_offset File offset
 * @param size read size
 */
void setup_range_request_impl(CurlHandle& curl, std::size_t file_offset, std::size_t size);
}  // namespace kvikio::detail
