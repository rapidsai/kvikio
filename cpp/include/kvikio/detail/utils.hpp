/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cstddef>
#include <cstring>
#include <exception>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

#include <kvikio/error.hpp>
#include <kvikio/logger.hpp>

namespace kvikio::detail {

/**
 * @brief Lowercase and whitespace-trim a copy of an environment-variable value.
 *
 * Used by the env-var parsers to accept case-insensitive values with optional surrounding
 * whitespace. The original string is left unchanged.
 *
 * @param s The raw environment-variable value (as returned by `std::getenv`).
 * @return A new `std::string` containing the lowercased, trimmed value.
 */
[[nodiscard]] std::string normalize_env_value(std::string_view s);

/**
 * @brief Round up `value` to multiples of `alignment`
 *
 * @param value Value to be rounded up
 * @param alignment Must be a power of 2
 * @return Rounded result
 */
[[nodiscard]] std::size_t align_up(std::size_t value, std::size_t alignment);

/**
 * @brief Align the address `addr` up to the boundary of `alignment`
 *
 * @param addr Address to be aligned up
 * @param alignment Must be a power of 2
 * @return Aligned address
 */
[[nodiscard]] void* align_up(void* addr, std::size_t alignment);

/**
 * @brief Round down `value` to multiples of `alignment`
 *
 * @param value Value to be rounded down
 * @param alignment Must be a power of 2
 * @return Rounded result
 */
[[nodiscard]] std::size_t align_down(std::size_t value, std::size_t alignment);

/**
 * @brief Align the address `addr` down to the boundary of `alignment`
 *
 * @param addr Address to be aligned down
 * @param alignment Must be a power of 2
 * @return Aligned address
 */
[[nodiscard]] void* align_down(void* addr, std::size_t alignment);

/**
 * @brief Whether `value` is a multiple of `alignment`
 *
 * @param value Value to be checked
 * @param alignment Must be a power of 2
 * @return Boolean answer
 */
bool is_aligned(std::size_t value, std::size_t alignment);

/**
 * @brief Whether the address `addr` is a multiple of `alignment`
 *
 * @param addr Address to be checked
 * @param alignment Must be a power of 2
 * @return Boolean answer
 */
bool is_aligned(void* addr, std::size_t alignment);

/**
 * @brief Prevent the compiler from optimizing away the read of a byte from a given address
 *
 * @param addr The address to read from
 */
void disable_read_optimization(void* addr);

/**
 * @brief Change an address `p` by a signed difference of `v`
 *
 * @tparam Integer Signed integer type
 * @param p An address
 * @param v Change of `p` in bytes
 * @return A new address as a result of applying `v` on `p`
 *
 * @note Technically, if the initial pointer is non-null, or does not point to an element of an
 * array object, (p + v) is undefined behavior (https://eel.is/c++draft/expr.add#4). However,
 * (p + v) on dynamic allocation is generally acceptable in practice, as long as users guarantee
 * that the resulting pointer points to a valid region.
 */
template <typename Integer>
void* pointer_add(void* p, Integer v)
{
  static_assert(std::is_integral_v<Integer>);
  return static_cast<std::byte*>(p) + v;
}

/**
 * @brief The distance in bytes between pointer `p1` and `p2`
 *
 * @param p1 The first pointer
 * @param p2 The second pointer
 * @return Signed result of (`p1` - `p2`). Both pointers are cast to std::byte* before subtraction.
 *
 * @note Technically, if two pointers do not point to elements from the same array, (p1 - p2) is
 * undefined behavior (https://eel.is/c++draft/expr.add#5). However, (p1 - p2) on dynamic allocation
 * is generally acceptable in practice, as long as users guarantee that both pointers are within the
 * valid region.
 */
std::ptrdiff_t pointer_diff(void* p1, void* p2);

/**
 * @brief For the specified memory range, touch the first byte of each page to cause page fault.
 *
 * For the first page, if the starting address is not aligned to the page boundary, the byte at
 * that address is touched.
 *
 * Reading is enough to fault in the pages of a file mapping. Anonymous memory that was never
 * written needs a write, because reading it only maps the shared zero page.
 *
 * @param buf The starting memory address
 * @param size The size in bytes of the memory range
 * @param write Whether to write zero to the touched byte of each page instead of reading it
 * @return The number of bytes touched
 */
std::size_t perform_prefault(void* buf, std::size_t size, bool write = false);

/**
 * @brief A simple scope guard that invokes a cleanup callable upon destruction.
 *
 * Guarantees the cleanup action runs when the guard goes out of scope, regardless of how the scope
 * is exited (normal return or exception). If the cleanup itself throws an exception, it is
 * suppressed to avoid calling std::terminate.
 *
 * Usage:
 * @code
 *   detail::ScopeExit guard([&]() { release(resource); });
 *   // Or
 *   auto guard = ScopeExit([&]() { release(resource); });
 * @endcode
 *
 * @tparam F A callable type invocable with no arguments.
 */
template <typename F>
class ScopeExit {
  static_assert(std::is_invocable_v<F>, "ScopeExit callable must be invocable with no arguments");

 private:
  F _cleanup;

 public:
  /**
   * @brief Constructs a scope guard that will invoke @p cleanup on destruction.
   * @param cleanup The cleanup callable to invoke on destruction
   */
  [[nodiscard]] explicit ScopeExit(F&& cleanup) : _cleanup(std::move(cleanup)) {}

  ~ScopeExit() noexcept
  {
    try {
      _cleanup();
    } catch (std::exception const& e) {
      KVIKIO_LOG_ERROR(e.what());
    } catch (...) {
      KVIKIO_LOG_ERROR("Unhandled exception");
    }
  }

  ScopeExit(ScopeExit const&)            = delete;
  ScopeExit& operator=(ScopeExit const&) = delete;
  ScopeExit(ScopeExit&&)                 = delete;
  ScopeExit& operator=(ScopeExit&&)      = delete;
};

}  // namespace kvikio::detail
