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

namespace {

/// Hand out the next observation id. From 1, so that a default-constructed `Observation` (id 0) is
/// never mistaken for a real one.
std::uint64_t next_observation_id() noexcept
{
  static std::atomic<std::uint64_t> id_counter{1};
  return id_counter.fetch_add(1, std::memory_order_relaxed);
}

}  // namespace

LogicalObservationRecorder::LogicalObservationRecorder(IoBackend backend,
                                                       TransferDirection direction,
                                                       MemoryKind memory_kind,
                                                       std::size_t offset,
                                                       std::size_t size,
                                                       std::string_view source,
                                                       char const* http_method) noexcept
  : _active{monitoring_enabled(ObservationKind::LOGICAL)}
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
  _observation.backend     = backend;
  _observation.direction   = direction;
  _observation.memory_kind = memory_kind;
  _observation.offset      = offset;
  _observation.size        = size;
  _observation.http_method = http_method;
  _observation.source      = source;
  _observation.id          = next_observation_id();
  _observation.start       = now();
  notify_started(_observation);
}

void LogicalObservationRecorder::emit() noexcept
{
  _observation.end = now();
  notify_finished(_observation);
}

PhysicalObservationRecorder::PhysicalObservationRecorder(PhysicalObservationContext const& context,
                                                         std::size_t offset,
                                                         std::size_t size) noexcept
  : _active{monitoring_enabled(ObservationKind::PHYSICAL)}
{
  if (!_active) { return; }
  _observation.kind        = ObservationKind::PHYSICAL;
  _observation.backend     = context.backend;
  _observation.direction   = context.direction;
  _observation.memory_kind = context.memory_kind;
  _observation.parent_id   = context.parent_id;
  _observation.source      = context.source;
  _observation.http_method = context.http_method;
  _observation.offset      = offset;
  _observation.size        = size;
  _observation.id          = next_observation_id();
  _observation.start       = now();
  notify_started(_observation);
}

PhysicalObservationRecorder::~PhysicalObservationRecorder()
{
  if (!_active || _emitted) { return; }
  _emitted                       = true;
  _observation.ok                = false;
  _observation.bytes_transferred = 0;
  emit();
}

void PhysicalObservationRecorder::finish(std::size_t bytes_transferred) noexcept
{
  if (!_active || _emitted) { return; }
  _emitted                       = true;
  _observation.bytes_transferred = bytes_transferred;
  emit();
}

void PhysicalObservationRecorder::emit() noexcept
{
  _observation.end = now();
  notify_finished(_observation);
}

}  // namespace kvikio::detail
