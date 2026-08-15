/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <string_view>

#include <kvikio/shim/utils.hpp>

namespace KVIKIO_EXPORT kvikio {

/**
 * @brief The clock KvikIO timestamps observations with.
 */
using Clock = std::chrono::steady_clock;

/**
 * @brief A point in time on `Clock`.
 */
using TimePoint = Clock::time_point;

/**
 * @brief A length of time, in nanoseconds.
 */
using Duration = std::chrono::nanoseconds;

/**
 * @brief A reading of `Clock` and of the wall clock, taken together.
 *
 * `Clock` cannot be compared with anything outside this process, so an anchor is what relates an
 * observation to a log line, to another process, or to a profiler trace.
 */
struct ClockAnchor {
  /// The reading of `Clock`.
  TimePoint steady{};
  /// The reading of the wall clock, taken at the same moment.
  std::chrono::system_clock::time_point wall{};

  /**
   * @brief Read both clocks, one immediately after the other.
   *
   * @warning The readings are tens of nanoseconds apart, since the two clocks cannot be read at
   * once, so the anchor's offset is out by that much. Well below what a wall clock is worth
   * anyway, NTP agreeing between machines to microseconds at best.
   *
   * @return The pair of readings.
   */
  [[nodiscard]] static ClockAnchor now() noexcept;

  /**
   * @brief Convert a point on `Clock` to wall-clock time.
   *
   * @warning The wall clock can be stepped or slewed by NTP or an operator, so an
   * old anchor may no longer be valid. Take an anchor at each end of a long run
   * and compare them to detect an in-flight adjustment to the system time.
   *
   * @param time The point to convert.
   * @return The corresponding wall-clock time.
   */
  [[nodiscard]] std::chrono::system_clock::time_point to_wall_clock(TimePoint time) const noexcept
  {
    return wall + std::chrono::duration_cast<std::chrono::system_clock::duration>(time - steady);
  }
};

/**
 * @brief The I/O backend that carried out an operation.
 */
enum class IoBackend : std::uint8_t {
  POSIX = 0,    ///< POSIX `pread`/`pwrite`, including the compatibility-mode path.
  GDS,          ///< cuFile / GPUDirect Storage.
  MMAP,         ///< Memory-mapped file access.
  REMOTE_HTTP,  ///< Remote I/O over HTTP(S), including S3.
  REMOTE_HDFS   ///< Remote I/O over WebHDFS.
};

/**
 * @brief Number of `IoBackend` values.
 */
constexpr std::size_t num_io_backends = static_cast<std::size_t>(IoBackend::REMOTE_HDFS) + 1;

/**
 * @brief The direction of an I/O operation.
 */
enum class TransferDirection : std::uint8_t {
  READ = 0,  ///< Data moves from the file or endpoint into the buffer.
  WRITE      ///< Data moves from the buffer into the file or endpoint.
};

/**
 * @brief The kind of memory the caller's buffer lives in.
 */
enum class MemoryKind : std::uint8_t {
  HOST = 0,  ///< Host (CPU) memory.
  DEVICE     ///< Device (GPU) memory.
};

/**
 * @brief What layer an observation describes.
 */
enum class ObservationKind : std::uint8_t {
  LOGICAL = 0  ///< One user-facing call, such as one `FileHandle::pread()`.
};

/**
 * @brief Human-readable name of an I/O backend.
 *
 * @param backend The backend.
 * @return A static string such as `"POSIX"`.
 */
[[nodiscard]] std::string_view to_string(IoBackend backend) noexcept;

/**
 * @brief Human-readable name of a transfer direction.
 *
 * @param direction The direction.
 * @return A static string such as `"READ"`.
 */
[[nodiscard]] std::string_view to_string(TransferDirection direction) noexcept;

/**
 * @brief Human-readable name of a memory kind.
 *
 * @param memory_kind The memory kind.
 * @return A static string such as `"DEVICE"`.
 */
[[nodiscard]] std::string_view to_string(MemoryKind memory_kind) noexcept;

/**
 * @brief Human-readable name of an observation kind.
 *
 * @param kind The kind.
 * @return A static string such as `"LOGICAL"`.
 */
[[nodiscard]] std::string_view to_string(ObservationKind kind) noexcept;

/**
 * @brief One I/O operation, as observed by KvikIO.
 *
 * `[start, end)` covers the operation from issue to completion.
 */
struct Observation {
  /// When the operation started.
  TimePoint start{};
  /// When the operation finished.
  TimePoint end{};
  /// Byte offset into the file or remote object.
  std::size_t offset{};
  /// Number of bytes requested.
  std::size_t size{};
  /// Number of bytes actually transferred. Differs from `size` on a short read, and is zero for an
  /// operation that failed.
  std::size_t bytes_transferred{};
  /// Identifies this operation, uniquely within the process.
  std::uint64_t id{};

  /// HTTP method, e.g. `"GET"`. Null for local I/O.
  char const* http_method{nullptr};

  /// What layer this describes. Always `ObservationKind::LOGICAL` today.
  ObservationKind kind{ObservationKind::LOGICAL};
  /// The backend that carried out the operation.
  IoBackend backend{IoBackend::POSIX};
  /// The direction of the operation.
  TransferDirection direction{TransferDirection::READ};
  /// The kind of memory the caller's buffer lives in.
  MemoryKind memory_kind{MemoryKind::HOST};
  /// False if the operation failed.
  bool ok{true};

  /**
   * @brief How long the operation took.
   *
   * @return `end - start`, or zero if the span is degenerate.
   */
  [[nodiscard]] Duration duration() const noexcept
  {
    return end > start ? end - start : Duration::zero();
  }

  /**
   * @brief Throughput of this single operation.
   *
   * @warning Averaging this across operations does *not* give the throughput of the program. For
   * that, divide total bytes by a span of elapsed time.
   *
   * @return Bytes per second, or zero if the operation had no measurable duration.
   */
  [[nodiscard]] double bytes_per_sec() const noexcept
  {
    auto const seconds = std::chrono::duration<double>{duration()}.count();
    return seconds > 0.0 ? static_cast<double>(bytes_transferred) / seconds : 0.0;
  }
};

/**
 * @brief Watches operations, from the moment they start until they finish.
 *
 * Derive from this and register it to be told what KvikIO is doing. Two notifications per
 * operation: `on_start()` when it begins, carrying the record as it stands at submission, and
 * `on_finish()` when it ends, carrying the finished record.
 *
 * A monitor is told about user-facing calls: one `FileHandle::pread()` is one operation however
 * many reads KvikIO issued underneath.
 *
 * @note Not everything is reported:
 * - The cuFile asynchronous API (`FileHandle::read_async()`, `FileHandle::write_async()`) on a
 *   system with working GDS reports nothing. In compatibility mode those calls fall back to
 *   `read()`/`write()` and *are* reported, so the same program is seen differently depending on
 *   whether GDS is available.
 * - The batch API (`BatchHandle`) reports nothing.
 * - `RemoteHandle::pread()` into device memory finishes when the last pinned-to-device copy is
 *   *issued* rather than completed, so its span is slightly short. It is never too long.
 *
 * @code
 * // Reports how many KvikIO operations are in flight at any moment.
 * class QueueDepth : public kvikio::Monitor {
 *  public:
 *   [[nodiscard]] int depth() const noexcept { return _in_flight.load(); }
 *
 *  private:
 *   void on_start(kvikio::Observation const&) noexcept override { ++_in_flight; }
 *   void on_finish(kvikio::Observation const&) noexcept override { --_in_flight; }
 *
 *   std::atomic<int> _in_flight{0};
 * };
 *
 * QueueDepth gauge;
 * auto const id = kvikio::register_monitor(&gauge);
 * ...
 * kvikio::unregister_monitor(id);  // Waits, so `gauge` may now be destroyed.
 * @endcode
 *
 * Every operation that reports a start reports exactly one finish, on whichever thread does the
 * work.
 *
 * @warning A monitor runs inline with the I/O, on the thread performing it, on both the submission
 * and the completion path. Keep it light, make it thread-safe, and do not call back into KvikIO,
 * which throws `std::runtime_error`. Neither callback may throw.
 */
class Monitor {
 public:
  Monitor()                          = default;
  virtual ~Monitor()                 = default;
  Monitor(Monitor const&)            = delete;
  Monitor& operator=(Monitor const&) = delete;

  /**
   * @brief An operation has started.
   *
   * The observation is not finished: `end` and `bytes_transferred` are zero, and `ok` is `true`
   * only because nothing has failed yet. All three are set by the time `on_finish()` is called.
   *
   * @warning Runs inline with the I/O, on the thread performing it. Keep it light, make it
   * thread-safe, and do not call back into KvikIO, which throws `std::runtime_error`.
   *
   * @param observation The operation, as far as it is known. The reference is valid only for the
   * duration of this call. Copy what is needed later.
   */
  virtual void on_start(Observation const& observation) noexcept = 0;

  /**
   * @brief An operation has completed.
   *
   * A monitor registered after `observation.start` never saw the matching `on_start()`, and
   * should ignore such an operation.
   *
   * @warning Runs inline with the I/O, on the thread performing it. Keep it light, make it
   * thread-safe, and do not call back into KvikIO, which throws `std::runtime_error`.
   *
   * @param observation The completed operation. The reference is valid only for the duration of
   * this call. Copy what is needed later.
   */
  virtual void on_finish(Observation const& observation) noexcept = 0;
};

/**
 * @brief Register a monitor, which begins receiving both notifications.
 *
 * @param monitor The monitor. Not owned, and must outlive the registration.
 * @param kind Which observations it watches.
 * @return An id for `unregister_monitor()`.
 *
 * @exception std::invalid_argument if `monitor` is null.
 * @exception std::runtime_error if called from inside a monitor callback.
 */
[[nodiscard]] std::uint64_t register_monitor(Monitor* monitor,
                                             ObservationKind kind = ObservationKind::LOGICAL);

/**
 * @brief Unregister a monitor.
 *
 * Blocks until no thread is inside either callback, so the monitor may be destroyed once this
 * returns.
 *
 * @param id The id from `register_monitor()`.
 */
void unregister_monitor(std::uint64_t id);

}  // namespace KVIKIO_EXPORT kvikio
