/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <vector>

#include <kvikio/observation.hpp>
#include <kvikio/shim/utils.hpp>
#include <kvikio/statistics/counters.hpp>

/**
 * @brief KvikIO namespace.
 */
namespace KVIKIO_EXPORT kvikio {
namespace statistics {

/**
 * @brief Running totals of the I/O KvikIO has performed.
 *
 * Everything here describes *logical* operations: one `pread()` is one operation however many
 * reads KvikIO issued underneath, and its duration covers the call from issue to completion.
 */
struct Summary {
  /// When counting started, or was last reset.
  TimePoint start{};
  /// When the summary was read.
  TimePoint end{};

  /**
   * @brief What relates `start` and `end` to the wall clock.
   *
   * They are read from a monotonic clock, so nothing outside this process can interpret them. The
   * monitor takes one anchor when it is constructed and stamps every reading with it. See
   * `kvikio::ClockAnchor` for how far to trust it over a long run.
   */
  ClockAnchor anchor{};

  /// Number of user-facing operations.
  std::uint64_t num_ops{};
  /// Of those, how many were reads and how many were writes.
  std::uint64_t num_reads{};
  std::uint64_t num_writes{};
  /// Bytes the operations asked for.
  std::uint64_t bytes_requested{};
  /// Bytes actually transferred. Differs from `bytes_requested` on a short read.
  std::uint64_t bytes_transferred{};
  /// Of those, how many were read and how many were written.
  std::uint64_t bytes_read{};
  std::uint64_t bytes_written{};
  /// Number of operations that failed.
  std::uint64_t num_errors{};

  /**
   * @brief What one backend carried, for `Summary::by_backend`.
   *
   * Every operation belongs to exactly one backend, so these add up to the summary's own. `busy`
   * has no counterpart here: it is a union over wall time, which two backends running at once
   * would both claim, so per-backend unions would not sum to the total.
   */
  struct BackendTotals {
    std::uint64_t num_ops{};
    std::uint64_t bytes_transferred{};
    /// The durations added up. Divide the bytes by this for what one operation averaged.
    Duration total_duration{};
    std::uint64_t num_errors{};
  };

  /// What each backend carried, indexed by `IoBackend`. Compatibility mode decides per call
  /// whether a read reaches cuFile or falls back to POSIX, so this is where that shows.
  std::array<BackendTotals, num_io_backends> by_backend{};

  /**
   * @brief The work in the span that belongs to no single operation.
   *
   * The counters run for the life of the process, and this is the part of them that falls inside
   * the span.
   */
  Counters counters{};

  /// The operations' durations added up, every operation counted.
  Duration total_duration{};

  /**
   * @brief Wall-clock time during which at least one operation was in flight.
   *
   * The union of the operations' time spans: overlapping work is counted once, the gaps between
   * calls count as idle, and it never exceeds `wall()`. An operation still running when the
   * reading is taken counts for the time it has been running so far.
   *
   * @warning An approximation, in both directions: a report delivered later than it was stamped
   * can have an idle gap counted as busy, or busy time missed. Both need two threads and a report
   * delayed past a whole operation. Against an exact merge of every span, the error is under
   * 1.5 % for 8 B reads on 8 threads and zero for a `pread()` of 4 KiB or more. Either way
   * `busy <= wall()` is guaranteed.
   */
  Duration busy{};

  /**
   * @brief Wall-clock span this summary covers.
   *
   * @return `end - start`, or zero if the span is degenerate.
   */
  [[nodiscard]] Duration wall() const noexcept;

  /**
   * @brief Throughput while KvikIO was actually busy.
   *
   * Understates while an operation is in flight, since its time counts from the moment it starts
   * and its bytes only once it completes.
   *
   * @return Bytes per second while busy, or zero if no time was spent busy.
   */
  [[nodiscard]] double busy_bytes_per_sec() const noexcept;

  /**
   * @brief Fraction of the span during which KvikIO was doing something.
   *
   * @return The ratio, or zero if the span is degenerate.
   */
  [[nodiscard]] double busy_fraction() const noexcept;

  /**
   * @brief Average time one operation took.
   *
   * @return `total_duration / num_ops`, or zero if nothing completed.
   */
  [[nodiscard]] Duration mean_duration() const noexcept;

  /**
   * @brief Totals for the interval between an earlier reading and this one.
   *
   * Every field is the difference of the two readings. `total_duration` is the exception: an
   * operation counts whole in the interval it finished in, however long it had been running.
   *
   * @param previous An earlier reading of the same span.
   * @exception std::invalid_argument if `previous` is not one, which covers an interval from
   * `since()`, a reading from another monitor, and one from before a `reset()`.
   * @return The interval's totals.
   */
  [[nodiscard]] Summary since(Summary const& previous) const;

  /**
   * @brief Serialize to JSON.
   *
   * The timestamps are against the wall clock, so another program can line the summary up with
   * its own log.
   *
   * @return A JSON object as a string.
   */
  [[nodiscard]] std::string to_json() const;

  /**
   * @brief Serialize to bytes, exactly.
   *
   * Everything survives, including the clock anchor and the monotonic timestamps, so a summary
   * that has been through a pipe is still a valid `previous` for `SummaryMonitor::since()`.
   *
   * @warning Not a wire format. The payload is this build's representation of the struct, so the
   * bytes are readable only by the same architecture and the same version of KvikIO. The header
   * carries the version, the size and a byte-order mark, so `deserialize()` refuses anything it
   * does not recognise and a mismatch is an exception rather than a wrong number. `to_json()` is
   * the format to use when something else has to read it.
   *
   * @return The bytes, the same number of them for every summary.
   */
  [[nodiscard]] std::vector<std::byte> serialize() const;

  /**
   * @brief Rebuild a summary from `serialize()`.
   *
   * @param bytes What `serialize()` produced, on this architecture and this build.
   * @exception std::invalid_argument if the bytes are not a summary, are the wrong length, or come
   * from a build whose summary differs from this one's.
   * @return The summary.
   */
  [[nodiscard]] static Summary deserialize(std::vector<std::byte> const& bytes);

  /**
   * @brief Format a human-readable report of every field.
   *
   * Byte counts, durations and rates are scaled to readable units. Use `to_json()` instead when
   * the output is going to be parsed.
   *
   * @code
   * KvikIO I/O summary
   *   wall time              243.80 ms
   *   operations             7
   *   ...
   * @endcode
   *
   * @param rows Which rows to print. Under `ReportRows::USED` a backend the run never reached and a
   * counter group it never touched are left out.
   * @return The report, one field per line, newline-terminated.
   */
  [[nodiscard]] std::string report(ReportRows rows = ReportRows::USED) const;
};

/**
 * @brief Turns on I/O statistics for the process and accumulates them while it exists.
 *
 * Create one early, keep it, and read it whenever a report is wanted:
 *
 * @code
 * kvikio::statistics::SummaryMonitor const monitor;  // statistics are now on
 * ...
 * auto const s = monitor.get();                      // totals so far
 * std::cout << s.bytes_transferred << " B, " << s.busy_bytes_per_sec() / 1e9 << " GB/s\n";
 * @endcode
 *
 * For a rate over an interval rather than since the beginning, difference two readings. The
 * result spans `[previous reading, now)`:
 *
 * @code
 * auto const before = monitor.get();
 * run_a_phase();
 * std::cout << monitor.since(before).busy_bytes_per_sec() / 1e9 << " GB/s during that phase\n";
 * @endcode
 *
 * Or hand it a callback and let it report itself once, when it goes out of scope:
 *
 * @code
 * kvikio::statistics::SummaryMonitor const monitor{[](kvikio::statistics::Summary const& s) {
 *   std::cout << s.bytes_transferred << " bytes\n";
 * }};
 * @endcode
 *
 * Monitors are independent. Any number can exist at once, nested or overlapping, and resetting
 * one has no effect on the others. An operation already in flight when the monitor is created is
 * ignored entirely, neither counted nor timed.
 *
 * @warning A monitor measures the whole process, not a scope. It counts every thread's I/O
 * while it exists, not only the I/O of the thread that created it, and it cannot attribute I/O to
 * a particular call. Wrapping a block in a monitor therefore measures that block only in a
 * program where nothing else is doing I/O at the same time. In a library, or anywhere with a
 * background reader, it will also count work you did not write.
 *
 * @warning The totals cover only what KvikIO observes, which is not every call it serves. See
 * `kvikio::Monitor` for what is left out. A program doing all its I/O that way reports zero
 * operations.
 *
 * Thread-safe: `get()`, `reset()` and `stop()` may be called from any thread while I/O is in
 * flight.
 *
 * ### Overhead
 *
 * Monitoring adds roughly 80 ns per logical operation, regardless of its size, so the relative
 * cost falls as the call grows: about 2 % of a 4 KiB `pread()` and 0.25 % of a 1 MiB one. With no
 * monitor registered it is about 5 ns per call.
 */
class SummaryMonitor final : private kvikio::Monitor {
 public:
  /// Called with the final totals when the monitor is destroyed.
  using Callback = std::function<void(Summary const&)>;

  /**
   * @brief Create a monitor and begin counting.
   */
  SummaryMonitor();

  /**
   * @brief Create a monitor that reports itself when it goes out of scope.
   *
   * @param on_destruction Invoked with the totals from the destructor. Exceptions it throws are
   * caught and logged, since a destructor cannot propagate them.
   */
  explicit SummaryMonitor(Callback on_destruction);

  /**
   * @brief Stop counting, invoke the callback if there is one, and release the registration.
   *
   * Waits for any in-flight observation delivery to finish, so the monitor's state is never
   * touched after this returns.
   */
  ~SummaryMonitor();

  // Not copyable or movable: the registry holds this object's address.
  SummaryMonitor(SummaryMonitor const&)            = delete;
  SummaryMonitor& operator=(SummaryMonitor const&) = delete;
  SummaryMonitor(SummaryMonitor&&)                 = delete;
  SummaryMonitor& operator=(SummaryMonitor&&)      = delete;

  /**
   * @brief Read the totals accumulated since construction, or since the last `reset()`.
   *
   * Safe to call repeatedly, and non-destructive.
   *
   * @warning This is a snapshot of a moving target. Operations still in flight are not included,
   * and neither are completed ones whose observation has not been delivered yet, so a reading
   * taken immediately after a single operation may fall short. Nothing is lost. It lands in a
   * later reading.
   *
   * @return The totals.
   */
  [[nodiscard]] Summary get() const;

  /**
   * @brief Zero the totals and restart the wall-clock span, as if the monitor had just been
   * constructed.
   *
   * @note On a monitor that has been stopped this leaves an empty summary over a degenerate span:
   * the origin moves forward, the end stays where `stop()` fixed it, and nothing further can be
   * counted.
   */
  void reset();

  /**
   * @brief Totals for the interval between an earlier reading and one taken now.
   *
   * `get().since(previous)`, for the common case of reporting one interval.
   *
   * @code
   * auto const before = monitor.get();
   * run_a_phase();
   * std::cout << monitor.since(before).busy_bytes_per_sec() << " B/s during that phase\n";
   * @endcode
   *
   * Reporting periodically wants one reading per tick, differenced against the last, rather than
   * this. Two calls would leave a gap between them, and an operation that completed in the gap
   * would fall into both intervals.
   *
   * @code
   * auto baseline = monitor.get();
   * while (running) {
   *   sleep_for(interval);
   *   auto const now = monitor.get();
   *   report(now.since(baseline));
   *   baseline = now;
   * }
   * @endcode
   *
   * @param previous An earlier reading from `get()` on this monitor, taken since the last
   * `reset()`.
   * @exception std::invalid_argument if `previous` is not one. See `Summary::since()`.
   * @return The interval's totals.
   */
  [[nodiscard]] Summary since(Summary const& previous) const;

  /**
   * @brief Stop counting. Idempotent, and one-way, there is no resuming.
   *
   * Safe to call from more than one thread, where the totals are final once any of them returns.
   *
   * The totals are final once this returns, and `get()` keeps returning them. The measured span
   * ends here too, so `wall()` and `busy_fraction()` describe the interval that was measured and
   * do not drift as the process goes on to do other things.
   */
  void stop();

 private:
  /// @brief The `Monitor` contract.
  void on_start(Observation const& observation) noexcept override;
  void on_finish(Observation const& observation) noexcept override;

  mutable std::mutex _mutex;

  /// Serializes `stop()`, which cannot use `_mutex`, since `unregister_monitor()` waits for
  /// notifications that take it.
  std::mutex _stopping;

  /// The internal counters as they stood when the span began, so the reading is a difference,
  /// and as they stood when it ended, so a stopped summary does not keep growing.
  Counters _counters_at_start{};
  Counters _counters_at_stop{};

  /**
   * @brief What every reading is a copy of.
   *
   * `Summary::end` and `Summary::busy` are the exceptions, since they belong to the reading rather
   * than to the totals, and `get()` fills them in.
   */
  Summary _totals{};

  /**
   * @brief When this monitor registered.
   *
   * Notifications go to whoever is registered at the time, which for a completion need not be who
   * was registered at the start. An operation that began before this instant is not ours. Until
   * the constructor sets it, no operation is.
   */
  TimePoint _registered{TimePoint::max()};

  /// When counting stopped, or the default while it is still running. Fixes the end of the measured
  /// span.
  TimePoint _stopped_end{};

  /// Registration with the observation facility, or 0 once stopped. Guarded by `_stopping`.
  std::uint64_t _registration{0};

  /// Handed the final totals by the destructor, or empty.
  Callback _on_destruction;

  /**
   * @brief How much of the span had at least one operation in flight.
   *
   * Busy time is the union of the operations' spans. Overlapping work counts once, and the gaps
   * between calls do not count at all. It is what `Summary::busy` holds, and what the busy
   * bandwidth divides by.
   *
   * The tracker follows one number, the count of operations in flight. A stretch of busy time
   * opens when that count rises off zero and closes when it returns to zero, so the union is never
   * assembled from the individual spans.
   *
   * The difficulty is that starts and completions arrive in the order threads reach the monitor's
   * lock, which need not be the order of the timestamps they carry. Three rules keep the total
   * sound:
   *
   * - A stretch closes at the latest end among its operations, rather than at whichever completion
   *   arrived last, which would drop the tail between the two.
   * - A stretch opens no earlier than the previous one closed. A start can arrive after an
   *   operation that began later has already completed, and reopening over time that has been
   *   counted would count the overlap twice.
   * - A reading never returns less than an earlier one did. Reading an open stretch runs it up to
   *   the instant of the reading, which a completion reported afterwards can revise downward, and
   *   the interval between the two readings would then be negative. The cost is that the reading
   *   which extrapolated keeps what it claimed, bounded by how late the completion was.
   *
   * Not thread safe. The monitor calls it while holding its own lock.
   */
  class BusyTracker {
   public:
    /// Begin a span at `start`, dropping everything counted before it. Operations already in
    /// flight keep their place and contribute only the part of their span that follows.
    void reset(TimePoint start);

    /// Report an operation that began at `start`.
    void on_start(TimePoint start);

    /// Report an operation that ended at `end`.
    void on_finish(TimePoint end);

    /// Busy time as of `now`, including the stretch still open.
    [[nodiscard]] Duration read(TimePoint now);

   private:
    /// Busy time from the stretches that have closed.
    Duration _closed_busy{};

    /// Operations started and not yet completed. A stretch is open while this is non-zero.
    std::uint64_t _in_flight{0};

    /// When the open stretch began. Meaningless while `_in_flight` is zero.
    TimePoint _busy_since{};

    /// The latest end seen among the operations of the open stretch.
    TimePoint _pending_end{};

    /// End of the most recently closed stretch, and the floor for the next one to open at.
    TimePoint _last_closed{};

    /// The largest busy time `read()` has returned.
    Duration _emitted_busy{};
  };

  /// Mutable because a reading advances the tracker's high-water mark.
  mutable BusyTracker _busy;
};

}  // namespace statistics
}  // namespace KVIKIO_EXPORT kvikio
