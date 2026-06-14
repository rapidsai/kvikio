/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <atomic>
#include <cstddef>

#include <kvikio/detail/concurrent_request_limiter.hpp>
#include <kvikio/error.hpp>
#include <kvikio/logger.hpp>
#include <kvikio/logger_macros.hpp>

namespace kvikio::detail {

ConcurrentRequestLimiter::ConcurrentRequestLimiter(std::size_t max_concurrent_requests) noexcept
  : _max_concurrent_requests{max_concurrent_requests}
{
}

bool ConcurrentRequestLimiter::try_acquire() noexcept
{
  if (_max_concurrent_requests == 0) {
    _count.fetch_add(1, std::memory_order_relaxed);
    return true;
  }
  // Increment only while strictly below the ceiling. A CAS loop keeps `_count` a hard cap, unlike a
  // plain fetch_add followed by a corrective fetch_sub, which would transiently expose an
  // over-count to concurrent acquirers.
  auto cur = _count.load(std::memory_order_relaxed);
  do {
    if (cur >= _max_concurrent_requests) { return false; }
  } while (!_count.compare_exchange_weak(
    cur, cur + 1, std::memory_order_relaxed, std::memory_order_relaxed));
  return true;
}

void ConcurrentRequestLimiter::release() noexcept
{
  [[maybe_unused]] auto const prev = _count.fetch_sub(1, std::memory_order_relaxed);

  // A release without a matching successful acquire may underflow the unsigned count and silently
  // disables the cap
  try {
    KVIKIO_EXPECT(prev > 0,
                  "ConcurrentRequestLimiter::release() called more times than try_acquire()");
  } catch (std::exception const& e) {
    KVIKIO_LOG_ERROR(e.what());
  }
}

bool ConcurrentRequestLimiter::unlimited() const noexcept { return _max_concurrent_requests == 0; }

}  // namespace kvikio::detail
