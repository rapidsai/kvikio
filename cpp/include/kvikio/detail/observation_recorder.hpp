/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>

#include <kvikio/observation.hpp>

namespace kvikio::detail {

/**
 * @brief Throw if the calling thread is inside a monitor callback.
 *
 * @exception std::runtime_error if called from inside a monitor callback.
 */
void expect_not_in_monitor();

/**
 * @brief The reading KvikIO timestamps observations with.
 *
 * @return Now, on `kvikio::Clock`.
 */
[[nodiscard]] inline TimePoint now() noexcept { return Clock::now(); }

/**
 * @brief Tell every monitor subscribed to its kind that an operation started.
 *
 * Runs the callbacks inline, on the calling thread. Does nothing if no monitor is registered, or
 * if the calling thread is already inside a monitor callback.
 *
 * @param observation The operation, with `start` set and the fields known at submission.
 */
void notify_started(Observation const& observation) noexcept;

/**
 * @brief Tell every monitor subscribed to its kind that an operation finished.
 *
 * Runs the callbacks inline, on the thread that completed the operation, which need not be the one
 * that started it. Does nothing if no monitor is registered, or if the calling thread is already
 * inside a monitor callback.
 *
 * @param observation The completed operation, with `end`, `bytes_transferred` and `ok` set.
 */
void notify_finished(Observation const& observation) noexcept;

/**
 * @brief Whether any monitor is registered.
 *
 * @return True if an observation would reach somebody.
 */
[[nodiscard]] bool monitoring_enabled() noexcept;

/**
 * @brief Times one user-facing call and emits its logical observation.
 *
 * Construction takes the start timestamp. `finish()` takes the end timestamp and notifies, exactly
 * once. There are two ways to use it, according to where the call actually ends.
 *
 * **Scope-bound**, for a blocking call: leave it on the stack and let the destructor finish it. An
 * exception leaving the scope marks the observation failed, so error paths are reported without
 * every call site having to catch.
 *
 * **Shared**, for a call whose work outlives the function that started it. `pread()` returns as
 * soon as the parts are submitted, so a destructor at the end of that function would stop the clock
 * before the work finished. Hold the recorder by `std::shared_ptr`, hand it to the tasks, and have
 * the task that completes the work call `finish()`. The end timestamp is then the true completion
 * time, the observation is delivered before the caller's future becomes ready, and none of it
 * depends on the caller ever waiting. A failing part calls `finish_with_failure()`, so one failed
 * part is one failed logical operation rather than a lost one.
 *
 * `finish()` is idempotent, so the destructor is a safe backstop under either use. When nobody is
 * subscribed the whole object collapses to a single relaxed atomic load. A shared user should still
 * test `monitoring_enabled()` first, to skip the allocation.
 */
class LogicalObservationRecorder {
 public:
  /**
   * @brief Start recording a call.
   *
   * @param backend The backend.
   * @param direction The direction.
   * @param memory_kind The kind of memory the caller's buffer lives in.
   * @param offset Byte offset into the file or remote object.
   * @param size Number of bytes requested.
   * @param http_method HTTP method for a remote call, e.g. `"GET"`. A constructor parameter rather
   * than a setter because the monitors are told the operation has started before this constructor
   * returns, and the record they are shown must be as complete as it can be.
   */
  LogicalObservationRecorder(IoBackend backend,
                             TransferDirection direction,
                             MemoryKind memory_kind,
                             std::size_t offset,
                             std::size_t size,
                             char const* http_method = nullptr) noexcept
    : _active{monitoring_enabled()}
  {
    if (_active) { begin(backend, direction, memory_kind, offset, size, http_method); }
  }

  /**
   * @brief Backstop. An operation that was never finished did not complete, so it is recorded as
   * a failure.
   *
   * Every path that completes the work calls `finish()` or `finish_with_failure()`, and this then
   * does nothing. Reaching here un-emitted means the call threw on its way to the work, or was
   * abandoned.
   */
  ~LogicalObservationRecorder() { finish_with_failure(); }

  LogicalObservationRecorder(LogicalObservationRecorder const&)            = delete;
  LogicalObservationRecorder& operator=(LogicalObservationRecorder const&) = delete;
  LogicalObservationRecorder(LogicalObservationRecorder&&)                 = delete;
  LogicalObservationRecorder& operator=(LogicalObservationRecorder&&)      = delete;

  /**
   * @brief The call completed. Stamp the end time and emit. Idempotent.
   *
   * @param bytes_transferred Number of bytes the call actually moved.
   */
  void finish(std::size_t bytes_transferred) noexcept
  {
    if (!_active) { return; }
    if (_emitted.exchange(true, std::memory_order_relaxed)) { return; }
    _observation.bytes_transferred = bytes_transferred;
    emit();
  }

  /**
   * @brief The call failed. Emit it as an error that moved nothing. Idempotent.
   *
   * One failing part fails the whole logical operation, so this is what a fan-out calls when any
   * of its parts failed.
   */
  void finish_with_failure() noexcept
  {
    if (!_active) { return; }
    if (_emitted.exchange(true, std::memory_order_relaxed)) { return; }
    _observation.ok                = false;
    _observation.bytes_transferred = 0;
    emit();
  }

 private:
  /// Fill in the record and tell the monitors. Out of line, see the constructor.
  void begin(IoBackend backend,
             TransferDirection direction,
             MemoryKind memory_kind,
             std::size_t offset,
             std::size_t size,
             char const* http_method) noexcept;

  /// Stamp the end and notify. Called once, from `finish()` or `finish_with_failure()`.
  void emit() noexcept;

  Observation _observation{};
  bool _active{false};
  std::atomic<bool> _emitted{false};
};

}  // namespace kvikio::detail
