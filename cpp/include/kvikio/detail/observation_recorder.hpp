/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string_view>

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
[[nodiscard]] TimePoint now() noexcept;

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
 * @brief Whether any monitor is subscribed to a kind of observation.
 *
 * @param kind The kind.
 * @return True if an observation of that kind would reach somebody.
 */
[[nodiscard]] bool monitoring_enabled(ObservationKind kind) noexcept;

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
 * `finish()` is idempotent, so the destructor is a safe backstop under either use. With nobody
 * subscribed the object does nothing beyond reading the monitor count, and fills in no record. A
 * shared user should still test `monitoring_enabled(ObservationKind::LOGICAL)` first, to skip
 * the allocation.
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
   * @param source The file path or URL, which must outlive the operation.
   * @param http_method HTTP method for a remote call, e.g. `"GET"`. Constructor parameters rather
   * than setters because the monitors are told the operation has started before this constructor
   * returns, and the record they are shown must be as complete as it can be.
   */
  LogicalObservationRecorder(IoBackend backend,
                             TransferDirection direction,
                             MemoryKind memory_kind,
                             std::size_t offset,
                             std::size_t size,
                             std::string_view source = {},
                             char const* http_method = nullptr) noexcept;

  /**
   * @brief Backstop. An operation that was never finished did not complete, so it is recorded as
   * a failure.
   *
   * Every path that completes the work calls `finish()` or `finish_with_failure()`, and this then
   * does nothing. Reaching here un-emitted means the call threw on its way to the work, or was
   * abandoned.
   */
  ~LogicalObservationRecorder();

  LogicalObservationRecorder(LogicalObservationRecorder const&)            = delete;
  LogicalObservationRecorder& operator=(LogicalObservationRecorder const&) = delete;
  LogicalObservationRecorder(LogicalObservationRecorder&&)                 = delete;
  LogicalObservationRecorder& operator=(LogicalObservationRecorder&&)      = delete;

  /**
   * @brief The call completed. Stamp the end time and emit. Idempotent.
   *
   * @param bytes_transferred Number of bytes the call actually moved.
   */
  void finish(std::size_t bytes_transferred) noexcept;

  /**
   * @brief The call failed. Emit it as an error that moved nothing. Idempotent.
   *
   * One failing part fails the whole logical operation, so this is what a fan-out calls when any
   * of its parts failed.
   */
  void finish_with_failure() noexcept;

  /**
   * @brief The id of the observation being recorded, for a physical observation to point at.
   *
   * @return The id, or empty if nobody is subscribed and no record is being filled in.
   */
  [[nodiscard]] std::optional<std::uint64_t> id() const noexcept
  {
    if (!_active) { return std::nullopt; }
    return _observation.id;
  }

 private:
  /// Fill in the record and tell the monitors. Out of line, see the constructor.
  void begin(IoBackend backend,
             TransferDirection direction,
             MemoryKind memory_kind,
             std::size_t offset,
             std::size_t size,
             std::string_view source,
             char const* http_method) noexcept;

  /// Stamp the end and notify. Called once, from `finish()` or `finish_with_failure()`.
  void emit() noexcept;

  Observation _observation{};
  bool _active{false};
  std::atomic<bool> _emitted{false};
};

/**
 * @brief What the physical observations of one call have in common.
 *
 * Built once by the call and copied into each task, which adds only its own offset and size.
 */
struct PhysicalObservationContext {
  /// The backend that carries out the transfers.
  IoBackend backend{IoBackend::POSIX};
  /// The direction of the transfers.
  TransferDirection direction{TransferDirection::READ};
  /// The kind of memory the caller's buffer lives in.
  MemoryKind memory_kind{MemoryKind::HOST};
  /// The id of the logical observation these transfers belong to. Empty when no monitor is
  /// registered for logical observations, since no logical record was made to point at.
  std::optional<std::uint64_t> parent_id{};
  /// The file path or URL, which must outlive the transfers.
  std::string_view source{};
  /// HTTP method for a remote transfer, e.g. `"GET"`.
  char const* http_method{nullptr};
};

/**
 * @brief Times one transfer and emits its physical observation.
 *
 * A transfer starts and finishes on one thread, so there is no shared mode. There are two ways to
 * use it, according to whether the transfer is a scope.
 *
 * **Scope-bound**, for a task body: leave it on the stack and let the destructor end it.
 *
 * **Owned by the transfer**, for a request the reactor drives across several of its loop
 * iterations: hold it in the object the request lives in, and destroy or replace it when the
 * attempt ends.
 *
 * Either way, destroying it without a `finish()` records the transfer as failed, so a task that
 * threw and a request that was abandoned are reported rather than lost.
 *
 * With nobody subscribed to `ObservationKind::PHYSICAL` the object does nothing beyond reading the
 * monitor count, and fills in no record.
 */
class PhysicalObservationRecorder {
 public:
  /**
   * @brief Start recording a transfer.
   *
   * @param context What this transfer shares with the others of the same call.
   * @param offset Byte offset into the file or remote object.
   * @param size Number of bytes this transfer moves.
   */
  PhysicalObservationRecorder(PhysicalObservationContext const& context,
                              std::size_t offset,
                              std::size_t size) noexcept;

  /**
   * @brief Backstop. A transfer that was never finished did not complete, so it is recorded as a
   * failure.
   */
  ~PhysicalObservationRecorder();

  PhysicalObservationRecorder(PhysicalObservationRecorder const&)            = delete;
  PhysicalObservationRecorder& operator=(PhysicalObservationRecorder const&) = delete;
  PhysicalObservationRecorder(PhysicalObservationRecorder&&)                 = delete;
  PhysicalObservationRecorder& operator=(PhysicalObservationRecorder&&)      = delete;

  /**
   * @brief The transfer completed. Stamp the end time and emit. Idempotent.
   *
   * @param bytes_transferred Number of bytes the transfer actually moved.
   */
  void finish(std::size_t bytes_transferred) noexcept;

 private:
  /// Stamp the end and notify.
  void emit() noexcept;

  Observation _observation{};
  bool _active{false};
  bool _emitted{false};
};

}  // namespace kvikio::detail
