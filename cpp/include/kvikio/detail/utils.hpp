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
 * @brief Round up `value` to multiples of `alignment`
 *
 * @param value Value to be rounded up
 * @param alignment Must be a power of 2
 * @return Rounded result
 */
[[nodiscard]] std::size_t align_up(std::size_t value, std::size_t alignment) noexcept;

/**
 * @brief Align the address `addr` up to the boundary of `alignment`
 *
 * @param addr Address to be aligned up
 * @param alignment Must be a power of 2
 * @return Aligned address
 */
[[nodiscard]] void* align_up(void* addr, std::size_t alignment) noexcept;

/**
 * @brief Round down `value` to multiples of `alignment`
 *
 * @param value Value to be rounded down
 * @param alignment Must be a power of 2
 * @return Rounded result
 */
[[nodiscard]] std::size_t align_down(std::size_t value, std::size_t alignment) noexcept;

/**
 * @brief Align the address `addr` down to the boundary of `alignment`
 *
 * @param addr Address to be aligned down
 * @param alignment Must be a power of 2
 * @return Aligned address
 */
[[nodiscard]] void* align_down(void* addr, std::size_t alignment) noexcept;

}  // namespace kvikio::detail
