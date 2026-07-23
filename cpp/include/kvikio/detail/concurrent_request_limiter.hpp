/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <atomic>
#include <cstddef>
#include <optional>

namespace kvikio::detail {

/**
 * @brief Non-blocking counting limiter that bounds the number of concurrent in-flight requests.
 *
 * Used by the `MULTI_POLL` remote I/O backend to cap the number of HTTP range requests that a
 * reactor keeps simultaneously attached to its multi handle. Without such a bound, one `pread()`
 * of a large file may submit too many requests which overwhelms the network. Each reactor owns its
 * own limiter, sized to a private share of the global budget
 * (`KVIKIO_REMOTE_IO_MAX_CONCURRENT_REQUESTS / num_reactors`).
 *
 * The limiter does not block. A caller that cannot acquire a slot is expected to defer its work
 * and retry later, so the reactor thread is free to keep driving the requests it already admitted.
 * `try_acquire()` is thread-safe, though in the current `MultiPollReactor` implementation all call
 * sites run on the same I/O thread.
 *
 * A reservation is represented by a move-only RAII `Slot`. The reservation returns to the limiter
 * when the `Slot` is destroyed.
 *
 * A `_max_concurrent_requests` of `std::nullopt` means unlimited.
 */
class ConcurrentRequestLimiter {
 public:
  /**
   * @brief Move-only RAII handle for one reserved limiter slot.
   *
   * An engaged `Slot` returns its reservation to the issuing limiter when destroyed, or earlier via
   * `reset()`. A default-constructed or moved-from `Slot` is empty and destroying it is a no-op.
   *
   */
  class Slot {
    friend class ConcurrentRequestLimiter;

   private:
    ConcurrentRequestLimiter* _limiter{nullptr};

    /**
     * @brief Construct an engaged slot bound to `limiter`.
     *
     * @param limiter The limiter the reservation was made on.
     */
    explicit Slot(ConcurrentRequestLimiter* limiter) noexcept;

   public:
    /**
     * @brief Construct an empty slot that holds no reservation.
     */
    Slot() noexcept = default;

    ~Slot() noexcept;

    // Move-only
    Slot(Slot const&)            = delete;
    Slot& operator=(Slot const&) = delete;
    Slot(Slot&& o) noexcept;
    Slot& operator=(Slot&& o) noexcept;

    /**
     * @brief Whether this object currently holds a reservation.
     */
    [[nodiscard]] explicit operator bool() const noexcept;

    /**
     * @brief Return the reservation to the limiter now instead of at destruction.
     */
    void reset() noexcept;
  };

  /**
   * @brief Construct a limiter with the given ceiling.
   *
   * @param max_concurrent_requests Maximum number of slots that may be held at once. `std::nullopt`
   * means unlimited.
   */
  explicit ConcurrentRequestLimiter(std::optional<std::size_t> max_concurrent_requests) noexcept;

  ConcurrentRequestLimiter(ConcurrentRequestLimiter const&)            = delete;
  ConcurrentRequestLimiter& operator=(ConcurrentRequestLimiter const&) = delete;
  ConcurrentRequestLimiter(ConcurrentRequestLimiter&&)                 = delete;
  ConcurrentRequestLimiter& operator=(ConcurrentRequestLimiter&&)      = delete;

  /**
   * @brief Try to reserve one slot without blocking.
   *
   * @return An engaged `Slot` holding the reservation, or an empty `Slot` if the limiter is already
   * at its ceiling. Always engaged when unlimited.
   */
  [[nodiscard]] Slot try_acquire() noexcept;

  /**
   * @brief Whether this limiter imposes no ceiling.
   */
  [[nodiscard]] bool unlimited() const noexcept;

 private:
  /**
   * @brief Return one previously reserved slot. Called by `Slot` only, which guarantees the
   * one-to-one pairing with a successful `try_acquire()`.
   */
  void release() noexcept;

  std::optional<std::size_t> const _max_concurrent_requests;  // std::nullopt means unlimited.
  std::atomic<std::size_t> _count{0};
};

}  // namespace kvikio::detail
