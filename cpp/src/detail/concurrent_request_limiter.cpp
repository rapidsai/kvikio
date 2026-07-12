/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <atomic>
#include <cstddef>
#include <optional>
#include <utility>

#include <kvikio/detail/concurrent_request_limiter.hpp>
#include <kvikio/logger.hpp>
#include <kvikio/logger_macros.hpp>

namespace kvikio::detail {

ConcurrentRequestLimiter::Slot::Slot(ConcurrentRequestLimiter* limiter) noexcept : _limiter{limiter}
{
}

ConcurrentRequestLimiter::Slot::~Slot() noexcept { reset(); }

ConcurrentRequestLimiter::Slot::Slot(Slot&& o) noexcept
  : _limiter{std::exchange(o._limiter, nullptr)}
{
}

ConcurrentRequestLimiter::Slot& ConcurrentRequestLimiter::Slot::operator=(Slot&& o) noexcept
{
  if (this != &o) {
    reset();
    _limiter = std::exchange(o._limiter, nullptr);
  }
  return *this;
}

ConcurrentRequestLimiter::Slot::operator bool() const noexcept { return _limiter != nullptr; }

void ConcurrentRequestLimiter::Slot::reset() noexcept
{
  if (_limiter != nullptr) {
    _limiter->release();
    _limiter = nullptr;
  }
}

ConcurrentRequestLimiter::ConcurrentRequestLimiter(
  std::optional<std::size_t> max_concurrent_requests) noexcept
  : _max_concurrent_requests{max_concurrent_requests}
{
}

ConcurrentRequestLimiter::Slot ConcurrentRequestLimiter::try_acquire() noexcept
{
  if (!_max_concurrent_requests.has_value()) {
    _count.fetch_add(1, std::memory_order_relaxed);
    return Slot{this};
  }
  // Increment only while strictly below the ceiling. A CAS loop keeps `_count` a hard cap, unlike a
  // plain fetch_add followed by a corrective fetch_sub, which would transiently expose an
  // over-count to concurrent acquirers.
  auto cur = _count.load(std::memory_order_relaxed);
  do {
    if (cur >= _max_concurrent_requests.value()) { return Slot{}; }
  } while (!_count.compare_exchange_weak(
    cur, cur + 1, std::memory_order_relaxed, std::memory_order_relaxed));
  return Slot{this};
}

void ConcurrentRequestLimiter::release() noexcept
{
  auto const prev = _count.fetch_sub(1, std::memory_order_relaxed);

  // This check is just a defense against future refactor errors.
  // A release without a matching successful acquire would underflow the unsigned count to SIZE_MAX,
  // after which try_acquire() refuses forever. `Slot` makes this impossible.
  if (prev == 0) {
    KVIKIO_LOG_ERROR("ConcurrentRequestLimiter::release() called more times than try_acquire()");
  }
}

bool ConcurrentRequestLimiter::unlimited() const noexcept
{
  return !_max_concurrent_requests.has_value();
}

}  // namespace kvikio::detail
