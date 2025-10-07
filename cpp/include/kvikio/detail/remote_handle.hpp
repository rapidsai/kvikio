/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include <cstring>

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
}  // namespace kvikio::detail
