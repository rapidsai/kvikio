/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
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
}  // namespace kvikio::detail
