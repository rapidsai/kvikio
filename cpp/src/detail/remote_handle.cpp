/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cstddef>
#include <stdexcept>
#include <string>

#include <curl/curl.h>

#include <kvikio/detail/remote_handle.hpp>
#include <kvikio/error.hpp>
#include <kvikio/remote_handle.hpp>
#include <kvikio/shim/libcurl.hpp>
#include <kvikio/statistics/counters.hpp>

namespace kvikio::detail {

std::size_t get_file_size_using_head_impl(RemoteEndpoint& endpoint)
{
  auto curl = create_curl_handle();

  endpoint.setopt(curl);
  curl.setopt(CURLOPT_NOBODY, 1L);
  curl.setopt(CURLOPT_FOLLOWLOCATION, 1L);
  {
    detail::ScopedTimer const probe{detail::count_remote_size_probe};
    curl.perform();
  }
  curl_off_t cl;
  curl.getinfo(CURLINFO_CONTENT_LENGTH_DOWNLOAD_T, &cl);
  KVIKIO_EXPECT(
    cl >= 0,
    "cannot get size of " + endpoint.str() + ", content-length not provided by the server",
    std::runtime_error);
  return static_cast<std::size_t>(cl);
}

void setup_range_request_impl(CurlHandle& curl, std::size_t file_offset, std::size_t size)
{
  KVIKIO_EXPECT(size > 0, "cannot create a zero-size range request", std::invalid_argument);
  std::string const byte_range =
    std::to_string(file_offset) + "-" + std::to_string(file_offset + size - 1);
  curl.setopt(CURLOPT_RANGE, byte_range.c_str());
}

}  // namespace kvikio::detail
