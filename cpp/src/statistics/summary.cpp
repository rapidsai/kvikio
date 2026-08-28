/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <locale>
#include <mutex>
#include <sstream>
#include <string>
#include <type_traits>
#include <utility>

#include <kvikio/detail/observation_recorder.hpp>
#include <kvikio/detail/string_utils.hpp>
#include <kvikio/error.hpp>
#include <kvikio/logger.hpp>
#include <kvikio/logger_macros.hpp>
#include <kvikio/observation.hpp>
#include <kvikio/statistics/summary.hpp>

namespace kvikio {
namespace statistics {

double Summary::busy_bytes_per_sec() const noexcept
{
  auto const seconds = std::chrono::duration<double>{busy}.count();
  return seconds > 0.0 ? static_cast<double>(bytes_transferred) / seconds : 0.0;
}

double Summary::busy_fraction() const noexcept
{
  auto const span = wall();
  if (span == Duration::zero()) { return 0.0; }
  return static_cast<double>(busy.count()) / static_cast<double>(span.count());
}

Duration Summary::wall() const noexcept { return end > start ? end - start : Duration::zero(); }

Duration Summary::mean_duration() const noexcept
{
  if (num_ops == 0) { return Duration::zero(); }
  return total_duration / static_cast<Duration::rep>(num_ops);
}

namespace {

/// "KVSM", a KvikIO summary, so that bytes from somewhere else are refused rather than read.
constexpr std::array<std::byte, 4> serial_magic{
  std::byte{'K'}, std::byte{'V'}, std::byte{'S'}, std::byte{'M'}};

/// Bumped whenever a field is added, removed or reordered. The size below catches most of that on
/// its own, but not two fields of the same width swapping places.
constexpr std::uint32_t serial_version = 2;

/// Written in the writer's byte order rather than in the header's, so that a reader whose order
/// differs sees it scrambled and refuses the payload instead of reinterpreting it.
constexpr std::uint32_t serial_byte_order = 0x01020304;

/// Where the byte-order mark sits, after the magic number, the version and the size.
constexpr std::size_t serial_byte_order_offset = 12;

/// The magic number, the version, the size and the byte-order mark that precede the object.
constexpr std::size_t serial_header_size = 16;

/// What `serialize()` produces and `deserialize()` requires.
constexpr std::size_t serialized_size = serial_header_size + sizeof(Summary);

/// Append a 32-bit header value, byte-order independent.
void put_u32(std::vector<std::byte>& out, std::uint32_t value)
{
  for (unsigned i = 0; i < 4; ++i) {
    out.push_back(static_cast<std::byte>((value >> (i * 8U)) & 0xFFU));
  }
}

/// Read a 32-bit header value, byte-order independent.
[[nodiscard]] std::uint32_t get_u32(std::vector<std::byte> const& in, std::size_t at)
{
  std::uint32_t value = 0;
  for (unsigned i = 0; i < 4; ++i) {
    value |= static_cast<std::uint32_t>(in[at + i]) << (i * 8U);
  }
  return value;
}

}  // namespace

// We `memcpy` the whole object, and `memcpy` is only defined for a trivially copyable type. This
// asks for more than that: it is false for a type with padding, whose indeterminate bytes would
// otherwise end up in the payload.
static_assert(std::has_unique_object_representations_v<Summary>);

std::vector<std::byte> Summary::serialize() const
{
  std::vector<std::byte> out;
  out.reserve(serialized_size);
  out.insert(out.end(), serial_magic.begin(), serial_magic.end());
  put_u32(out, serial_version);
  put_u32(out, static_cast<std::uint32_t>(sizeof(Summary)));
  out.resize(serialized_size);
  std::memcpy(out.data() + serial_byte_order_offset, &serial_byte_order, sizeof(serial_byte_order));
  std::memcpy(out.data() + serial_header_size, this, sizeof(Summary));
  return out;
}

Summary Summary::deserialize(std::vector<std::byte> const& bytes)
{
  KVIKIO_EXPECT(bytes.size() == serialized_size,
                "not a serialized kvikio::statistics::Summary: wrong length",
                std::invalid_argument);
  KVIKIO_EXPECT(std::equal(serial_magic.begin(), serial_magic.end(), bytes.begin()),
                "not a serialized kvikio::statistics::Summary: wrong magic",
                std::invalid_argument);

  auto const version = get_u32(bytes, 4);
  auto const size    = get_u32(bytes, 8);
  KVIKIO_EXPECT(version == serial_version && size == sizeof(Summary),
                "serialized kvikio::statistics::Summary from a different build of KvikIO, "
                "version " +
                  std::to_string(version) + " of " + std::to_string(size) +
                  " bytes against version " + std::to_string(serial_version) + " of " +
                  std::to_string(sizeof(Summary)),
                std::invalid_argument);

  std::uint32_t byte_order = 0;
  std::memcpy(&byte_order, bytes.data() + serial_byte_order_offset, sizeof(byte_order));
  KVIKIO_EXPECT(byte_order == serial_byte_order,
                "serialized kvikio::statistics::Summary written on a machine of a different byte "
                "order, which this format does not translate",
                std::invalid_argument);

  Summary ret;
  std::memcpy(&ret, bytes.data() + serial_header_size, sizeof(Summary));
  return ret;
}

std::string Summary::to_json() const
{
  std::ostringstream os;
  // The global locale can put a comma where JSON needs a point, and can group digits.
  os.imbue(std::locale::classic());
  // Wall clock, so a reader can line these up with anything else that is timestamped. The
  // measurement stays monotonic, see `Summary::anchor`.
  auto const wall_ns = [this](TimePoint time) {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
             anchor.to_wall_clock(time).time_since_epoch())
      .count();
  };
  os << "{\"kind\": \"" << to_string(kind) << "\", \"start_unix_ns\": " << wall_ns(start)
     << ", \"end_unix_ns\": " << wall_ns(end) << ", \"wall_ns\": " << wall().count()
     << ", \"num_ops\": " << num_ops << ", \"num_reads\": " << num_reads
     << ", \"num_writes\": " << num_writes << ", \"bytes_requested\": " << bytes_requested
     << ", \"bytes_transferred\": " << bytes_transferred << ", \"bytes_read\": " << bytes_read
     << ", \"bytes_written\": " << bytes_written << ", \"num_errors\": " << num_errors
     << ", \"busy_ns\": " << busy.count() << ", \"total_duration_ns\": " << total_duration.count()
     << ", \"busy_bytes_per_sec\": " << busy_bytes_per_sec()
     << ", \"busy_fraction\": " << busy_fraction()
     << ", \"mean_duration_ns\": " << mean_duration().count() << ", \"by_backend\": {";
  for (std::size_t i = 0; i < num_io_backends; ++i) {
    auto const& totals = by_backend[i];
    if (i != 0) { os << ", "; }
    os << "\"" << kvikio::to_string(static_cast<IoBackend>(i))
       << "\": {\"num_ops\": " << totals.num_ops
       << ", \"bytes_transferred\": " << totals.bytes_transferred
       << ", \"total_duration_ns\": " << totals.total_duration.count()
       << ", \"num_errors\": " << totals.num_errors << "}";
  }
  os << "}}";
  return os.str();
}

std::string Summary::report() const
{
  std::ostringstream os;
  // The width is that of the widest label below, so the values line up.
  auto const row = [&os](std::string const& label, auto const&... parts) {
    std::ostringstream value;
    (value << ... << parts);
    os << "  " << std::left << std::setw(21) << label << value.str() << "\n";
  };

  os << "KvikIO I/O summary (" << to_string(kind) << ")\n";
  row("wall time", detail::format_duration(wall()));
  // The share of the wall time is what makes the duration meaningful.
  row("busy time",
      detail::format_duration(busy),
      " (",
      std::fixed,
      std::setprecision(2),
      busy_fraction() * 100.0,
      " % of the wall time)");
  row("busy bandwidth", detail::format_rate(busy_bytes_per_sec()));
  row("operations", num_ops, " (", num_reads, " read, ", num_writes, " write)");
  // Not a latency: a duration covers moving all of the operation's bytes, so it grows with the
  // size of the read.
  row("mean duration", detail::format_duration(mean_duration()));
  // What arrived against what was asked for, so a shortfall needs no row of its own.
  row("bytes",
      detail::format_nbytes(bytes_transferred),
      " of ",
      detail::format_nbytes(bytes_requested),
      " requested (",
      detail::format_nbytes(bytes_read),
      " read, ",
      detail::format_nbytes(bytes_written),
      " written)");
  row("errors", num_errors);
  // Every backend gets a row, including the ones that carried nothing, since "no GDS here" is an
  // answer as often as the bytes are, and a report of a fixed shape can be compared with another.
  for (std::size_t i = 0; i < num_io_backends; ++i) {
    auto const& totals = by_backend[i];
    auto const label =
      std::string{"backend "} + std::string{kvikio::to_string(static_cast<IoBackend>(i))};
    if (totals.num_ops == 0) {
      row(label, "unused");
      continue;
    }
    std::ostringstream value;
    value << detail::format_nbytes(totals.bytes_transferred) << " in " << totals.num_ops << " ops, "
          << detail::format_duration(totals.total_duration);
    // What one operation averaged, which compares backends whatever concurrency each was given.
    if (totals.total_duration > Duration::zero()) {
      auto const seconds =
        std::chrono::duration_cast<std::chrono::duration<double>>(totals.total_duration).count();
      value << ", " << detail::format_rate(static_cast<double>(totals.bytes_transferred) / seconds);
    }
    if (totals.num_errors != 0) { value << ", " << totals.num_errors << " failed"; }
    row(label, value.str());
  }
  return os.str();
}

SummaryMonitor::SummaryMonitor(ObservationKind kind) : SummaryMonitor{Callback{}, kind} {}

SummaryMonitor::SummaryMonitor(Callback on_destruction, ObservationKind kind)
  : _on_destruction{std::move(on_destruction)}
{
  // Registration comes first, and the span is stamped after it, under the lock. `_registered`
  // starts at its maximum, so an operation that starts in between is refused by both callbacks
  // rather than by only one of them, which would leave the tracker's in-flight count unbalanced.
  _registration = register_monitor(this, kind);
  std::lock_guard const lock{_mutex};
  auto const t   = detail::now();
  _totals.start  = t;
  _totals.anchor = ClockAnchor::now();
  _totals.kind   = kind;
  _registered    = t;
  _busy.reset(t);
}

SummaryMonitor::~SummaryMonitor()
{
  // First, before any member is destroyed: it unregisters and waits for a notification in progress,
  // so nothing can be inside this object afterwards.
  stop();
  if (!_on_destruction) { return; }
  try {
    _on_destruction(get());
  } catch (std::exception const& e) {
    KVIKIO_LOG_ERROR(std::string("SummaryMonitor callback threw: ") + e.what());
  } catch (...) {
    KVIKIO_LOG_ERROR("SummaryMonitor callback threw an unknown exception");
  }
}

void SummaryMonitor::BusyTracker::reset(TimePoint start)
{
  _closed_busy  = Duration::zero();
  _emitted_busy = Duration::zero();
  _pending_end  = TimePoint{};
  // `_in_flight` is left alone, since the operations it counts still have to report their
  // completion. Re-anchoring the open stretch is what holds the total within the new span.
  _busy_since  = start;
  _last_closed = start;
}

void SummaryMonitor::BusyTracker::on_start(TimePoint start)
{
  // `_last_closed` keeps a start that arrives late from reopening time that has been counted.
  auto const from = std::max(start, _last_closed);
  // The first operation opens a stretch and a further one joins it, extending it backwards if it
  // began earlier.
  _busy_since = _in_flight++ == 0 ? from : std::min(_busy_since, from);
}

void SummaryMonitor::BusyTracker::on_finish(TimePoint end)
{
  if (_in_flight == 0) { return; }
  _pending_end = std::max(_pending_end, end);
  if (--_in_flight > 0) { return; }
  // The stretch ran until the last of its operations ended, which is not necessarily this one.
  if (_pending_end > _busy_since) { _closed_busy += _pending_end - _busy_since; }
  _last_closed = std::max(_last_closed, _pending_end);
  _pending_end = TimePoint{};
}

Duration SummaryMonitor::BusyTracker::read(TimePoint now)
{
  auto const open = (_in_flight > 0 && now > _busy_since) ? now - _busy_since : Duration::zero();
  _emitted_busy   = std::max(_emitted_busy, _closed_busy + open);
  return _emitted_busy;
}

void SummaryMonitor::on_start(Observation const& observation) noexcept
{
  auto const start = observation.start;
  std::lock_guard const lock{_mutex};
  if (start < _registered) { return; }
  _busy.on_start(start);
}

void SummaryMonitor::on_finish(Observation const& observation) noexcept
{
  std::lock_guard const lock{_mutex};
  // Not ours: it began before we were registered, so we never counted its start and its bytes
  // belong to a span we were not measuring.
  if (observation.start < _registered) { return; }

  // Close the operation first, and unconditionally, so that the tracker's in-flight count stays
  // balanced even for a record the counters below decline.
  _busy.on_finish(observation.end);

  // A `reset()` since the operation began has moved the span on, and the operation is not part of
  // it. Counting it would add bytes and a duration that ran before the span did, while `busy`
  // holds only the part that followed the reset.
  if (observation.start < _totals.start) { return; }

  ++_totals.num_ops;
  _totals.bytes_requested += observation.size;
  _totals.bytes_transferred += observation.bytes_transferred;
  _totals.total_duration += observation.duration();
  auto const backend = static_cast<std::size_t>(observation.backend);
  if (backend < num_io_backends) {
    auto& totals = _totals.by_backend[backend];
    ++totals.num_ops;
    totals.bytes_transferred += observation.bytes_transferred;
    totals.total_duration += observation.duration();
    if (!observation.ok) { ++totals.num_errors; }
  }
  if (observation.direction == TransferDirection::READ) {
    ++_totals.num_reads;
    _totals.bytes_read += observation.bytes_transferred;
  } else {
    ++_totals.num_writes;
    _totals.bytes_written += observation.bytes_transferred;
  }
  if (!observation.ok) { ++_totals.num_errors; }
}

Summary SummaryMonitor::get() const
{
  std::lock_guard const lock{_mutex};
  Summary ret = _totals;
  // Measured to the same instant the reading is stamped with, inside the lock, so that busy time
  // can never run past the end of the span it is divided by and `busy_fraction()` stays <= 1.
  ret.end  = _stopped_end != TimePoint{} ? _stopped_end : detail::now();
  ret.busy = _busy.read(ret.end);
  return ret;
}

Summary Summary::since(Summary const& previous) const
{
  // Every cumulative reading of a span begins where the span does, and an interval does not, so
  // this catches a caller feeding an interval back in as the next baseline, which would subtract
  // deltas from totals and count the earlier operations again. It catches a reading of a different
  // span, from another monitor or from before a `reset()`, for the same reason.
  KVIKIO_EXPECT(previous.start == start && previous.end <= end,
                "`previous` must be an earlier reading of the same span, from `get()` rather than "
                "from `since()`",
                std::invalid_argument);

  // Saturating, for the counters and the durations alike. A reading can be overtaken by a
  // completion reported after it, so a later one is not always the larger.
  auto const subtract = [](auto lhs, auto rhs) { return lhs > rhs ? lhs - rhs : decltype(lhs){}; };

  std::array<BackendTotals, num_io_backends> backends{};
  for (std::size_t i = 0; i < num_io_backends; ++i) {
    auto const& now = by_backend[i];
    auto const& was = previous.by_backend[i];
    backends[i]     = {.num_ops           = subtract(now.num_ops, was.num_ops),
                       .bytes_transferred = subtract(now.bytes_transferred, was.bytes_transferred),
                       .total_duration    = subtract(now.total_duration, was.total_duration),
                       .num_errors        = subtract(now.num_errors, was.num_errors)};
  }

  return Summary{.start             = previous.end,
                 .end               = end,
                 .anchor            = anchor,
                 .num_ops           = subtract(num_ops, previous.num_ops),
                 .num_reads         = subtract(num_reads, previous.num_reads),
                 .num_writes        = subtract(num_writes, previous.num_writes),
                 .bytes_requested   = subtract(bytes_requested, previous.bytes_requested),
                 .bytes_transferred = subtract(bytes_transferred, previous.bytes_transferred),
                 .bytes_read        = subtract(bytes_read, previous.bytes_read),
                 .bytes_written     = subtract(bytes_written, previous.bytes_written),
                 .num_errors        = subtract(num_errors, previous.num_errors),
                 .by_backend        = backends,
                 .total_duration    = subtract(total_duration, previous.total_duration),
                 .busy              = subtract(busy, previous.busy),
                 .kind              = kind};
}

Summary SummaryMonitor::since(Summary const& previous) const { return get().since(previous); }

void SummaryMonitor::reset()
{
  std::lock_guard const lock{_mutex};
  auto const t      = detail::now();
  auto const anchor = _totals.anchor;
  auto const kind   = _totals.kind;
  _totals           = Summary{};
  _totals.start     = t;
  _totals.anchor    = anchor;
  _totals.kind      = kind;
  // Keeps `busy <= wall()` across the reset: an operation already in flight contributes only the
  // part of its span that follows it.
  _busy.reset(t);
}

void SummaryMonitor::stop()
{
  std::lock_guard const stopping{_stopping};
  if (_registration == 0) { return; }
  // Waits for a notification in progress, so once this returns no thread is inside this object.
  unregister_monitor(_registration);
  _registration = 0;
  // Stamped after the drain, so every operation that was going to be counted ends at or before it.
  // From here on this is the end of the measured span, however long the monitor lives on.
  std::lock_guard const lock{_mutex};
  _stopped_end = detail::now();
}

}  // namespace statistics
}  // namespace kvikio
