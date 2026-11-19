/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cstddef>
#include <string>

#include <kvikio/remote_handle.hpp>

namespace kvikio::detail {

/**
 * @brief Get the file size, if using `HEAD` request to obtain the content-length header is
 * permitted.
 *
 * This function works for the `HttpEndpoint` and `S3Endpoint`, but not for
 * `S3EndpointWithPresignedUrl`, which does not allow `HEAD` request.
 *
 * @param endpoint The remote endpoint
 * @param url The URL of the remote file
 * @return The file size
 */
std::size_t get_file_size_using_head_impl(RemoteEndpoint& endpoint, std::string const& url);

/**
 * @brief Set up the range request for libcurl. Use this method when HTTP range request is supposed.
 *
 * @param curl A curl handle
 * @param file_offset File offset
 * @param size read size
 */
void setup_range_request_impl(CurlHandle& curl, std::size_t file_offset, std::size_t size);

}  // namespace kvikio::detail
