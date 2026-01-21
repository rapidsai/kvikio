/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cstring>

#include <kvikio/shim/libcurl.hpp>

namespace kvikio::detail {
/**
 * @brief Check a libcurl easy interface return code and throw on error.
 *
 * @param err_code The CURLcode to check.
 * @exception std::runtime_error if err_code is not CURLE_OK.
 */
#define KVIKIO_CHECK_CURL_EASY(err_code) \
  kvikio::detail::check_curl_easy(err_code, __FILE__, __LINE__)

/**
 * @brief Check a libcurl multi interface return code and throw on error.
 *
 * @param err_code The CURLMcode to check.
 * @exception std::runtime_error if err_code is not CURLM_OK.
 */
#define KVIKIO_CHECK_CURL_MULTI(err_code) \
  kvikio::detail::check_curl_multi(err_code, __FILE__, __LINE__)

/**
 * @brief Check a libcurl easy interface return code and throw on error.
 *
 * @param err_code The CURLcode to check.
 * @param filename Source filename for error reporting.
 * @param line_number Source line number for error reporting.
 * @exception std::runtime_error if err_code is not CURLE_OK.
 */
void check_curl_easy(CURLcode err_code, char const* filename, int line_number);

/**
 * @brief Check a libcurl multi interface return code and throw on error.
 *
 * @param err_code The CURLMcode to check.
 * @param filename Source filename for error reporting.
 * @param line_number Source line number for error reporting.
 * @exception std::runtime_error if err_code is not CURLM_OK.
 */
void check_curl_multi(CURLMcode err_code, char const* filename, int line_number);

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
