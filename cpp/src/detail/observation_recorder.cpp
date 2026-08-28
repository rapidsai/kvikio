/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <string_view>

#include <kvikio/detail/observation_recorder.hpp>
#include <kvikio/observation.hpp>

namespace kvikio::detail {

TimePoint now() noexcept { return Clock::now(); }

LogicalObservationRecorder::LogicalObservationRecorder(IoBackend backend,
                                                       TransferDirection direction,
                                                       MemoryKind memory_kind,
                                                       std::size_t offset,
                                                       std::size_t size,
                                                       std::string_view source,
                                                       char const* http_method) noexcept
  : _active{monitoring_enabled()}
{
  if (_active) { begin(backend, direction, memory_kind, offset, size, source, http_method); }
}

LogicalObservationRecorder::~LogicalObservationRecorder() { finish_with_failure(); }

void LogicalObservationRecorder::finish(std::size_t bytes_transferred) noexcept
{
  if (!_active) { return; }
  if (_emitted.exchange(true, std::memory_order_relaxed)) { return; }
  _observation.bytes_transferred = bytes_transferred;
  emit();
}

void LogicalObservationRecorder::finish_with_failure() noexcept
{
  if (!_active) { return; }
  if (_emitted.exchange(true, std::memory_order_relaxed)) { return; }
  _observation.ok                = false;
  _observation.bytes_transferred = 0;
  emit();
}

void LogicalObservationRecorder::begin(IoBackend backend,
                                       TransferDirection direction,
                                       MemoryKind memory_kind,
                                       std::size_t offset,
                                       std::size_t size,
                                       std::string_view source,
                                       char const* http_method) noexcept
{
  // From 1, so that a default-constructed `Observation` (id 0) is never mistaken for a real one.
  static std::atomic<std::uint64_t> id_counter{1};
  _observation.backend     = backend;
  _observation.direction   = direction;
  _observation.memory_kind = memory_kind;
  _observation.offset      = offset;
  _observation.size        = size;
  _observation.http_method = http_method;
  _observation.source      = source;
  _observation.id          = id_counter.fetch_add(1, std::memory_order_relaxed);
  _observation.start       = now();
  notify_started(_observation);
}

void LogicalObservationRecorder::emit() noexcept
{
  _observation.end = now();
  notify_finished(_observation);
}

}  // namespace kvikio::detail
