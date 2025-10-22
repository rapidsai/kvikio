/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
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
