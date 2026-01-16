/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <string>

#include <kvikio/detail/remote_handle.hpp>

namespace kvikio::detail {
std::size_t callback_get_string_response(char* data,
                                         std::size_t size,
                                         std::size_t num_bytes,
                                         void* userdata)
{
  auto new_data_size = size * num_bytes;
  auto* response     = reinterpret_cast<std::string*>(userdata);
  response->append(data, new_data_size);
  return new_data_size;
}

void setup_range_request_impl(CurlHandle& curl, std::size_t file_offset, std::size_t size)
{
  std::string const byte_range =
    std::to_string(file_offset) + "-" + std::to_string(file_offset + size - 1);
  curl.setopt(CURLOPT_RANGE, byte_range.c_str());
}
}  // namespace kvikio::detail
