/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cstring>
#include <exception>
#include <utility>

#include <kvikio/error.hpp>

namespace kvikio::detail {

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
