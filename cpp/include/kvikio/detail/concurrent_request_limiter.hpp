/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <atomic>
#include <cstddef>

namespace kvikio::detail {

/**
 * @brief Non-blocking counting limiter that bounds the number of concurrent in-flight requests.
 *
 * Used by the `MULTI_POLL` remote I/O backend to cap the number of HTTP range requests (libcurl
 * easy handles) that a reactor keeps simultaneously attached to its multi handle. Without such a
 * bound, one `pread()` of a large file fans out to one request per sub-range (e.g. thousands at
 * once), which overwhelms the network path and collapses throughput. Each `MultiPollReactor` owns
 * its own limiter, sized to a private share of the global budget
 * (`KVIKIO_REMOTE_IO_MAX_CONCURRENT_REQUESTS / num_reactors`); per-reactor shares avoid a single
 * shared admission gate, which would otherwise starve reactors and stall the pipeline under heavy
 * re-admission churn.
 *
 * The limiter never blocks. A caller that cannot acquire a slot is expected to defer its work and
 * retry later, so the reactor thread is free to keep driving the requests it already admitted
 * (which is what eventually frees slots). `try_acquire()` and `release()` are thread-safe so that
 * `release()` may run on a different thread than `try_acquire()`, for example a release from a
 * completion callback (used by the device-buffer path).
 *
 * A `_max` of 0 means unlimited. To keep the acquire/release pairing uniform at every call site,
 * the limiter still counts in the unlimited case (`try_acquire()` always succeeds), so callers
 * always pair a successful `try_acquire()` with exactly one `release()` regardless of
 * configuration.
 */
class ConcurrentRequestLimiter {
 public:
  /**
   * @brief Construct a limiter with the given ceiling.
   *
   * @param max_concurrent_requests Maximum number of slots that may be held at once. 0 means
   * unlimited.
   */
  explicit ConcurrentRequestLimiter(std::size_t max_concurrent_requests) noexcept;

  ConcurrentRequestLimiter(ConcurrentRequestLimiter const&)            = delete;
  ConcurrentRequestLimiter& operator=(ConcurrentRequestLimiter const&) = delete;
  ConcurrentRequestLimiter(ConcurrentRequestLimiter&&)                 = delete;
  ConcurrentRequestLimiter& operator=(ConcurrentRequestLimiter&&)      = delete;

  /**
   * @brief Try to reserve one slot without blocking.
   *
   * @return `true` if a slot was reserved (the caller must later call `release()` exactly once), or
   * `false` if the limiter is already at its ceiling. Always returns `true` when unlimited.
   */
  [[nodiscard]] bool try_acquire() noexcept;

  /**
   * @brief Return one previously reserved slot. Must be paired one-to-one with a successful
   * `try_acquire()`.
   */
  void release() noexcept;

  /**
   * @brief Whether this limiter imposes no ceiling.
   */
  [[nodiscard]] bool unlimited() const noexcept;

 private:
  std::size_t const _max;  // 0 means unlimited.
  std::atomic<std::size_t> _count{0};
};

}  // namespace kvikio::detail
