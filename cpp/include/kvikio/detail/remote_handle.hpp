/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cstddef>

#include <kvikio/remote_handle.hpp>

namespace kvikio::detail {

/**
 * @brief Get the file size from the `Content-Length` header of a `HEAD` request.
 *
 * This function works for an endpoint whose server answers a `HEAD` request with the file size. It
 * does not work for `S3EndpointWithPresignedUrl`, because AWS S3 does not allow a `HEAD` request
 * with a presigned URL.
 *
 * @param endpoint The remote endpoint. Its `setopt()` sets the URL and any headers of the request.
 * @return The file size in bytes.
 */
std::size_t get_file_size_using_head_impl(RemoteEndpoint& endpoint);

/**
 * @brief Set up a curl handle to read part of a file, with the range in a `Range` header. Use this
 * function when the server supports HTTP range requests.
 *
 * @param curl The curl handle.
 * @param file_offset The offset in the file of the first byte to read.
 * @param size The number of bytes to read.
 * @exception std::invalid_argument if `size` is 0.
 */
void setup_range_request_impl(CurlHandle& curl, std::size_t file_offset, std::size_t size);

}  // namespace kvikio::detail
