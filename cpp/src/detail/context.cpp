/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <kvikio/detail/context.hpp>
#include <kvikio/detail/event.hpp>
#include <kvikio/detail/nvtx.hpp>
#include <kvikio/error.hpp>
#include <kvikio/shim/cuda.hpp>

namespace kvikio::detail {

[[nodiscard]] CUcontext IoContext::cuda_context() const noexcept { return _cuda_context; }

void IoContext::record_event(CUstream stream)
{
  KVIKIO_NVTX_FUNC_RANGE();
  auto const tid = std::this_thread::get_id();

  std::lock_guard const lock(_mutex);

  // If not found, acquire an event from the pool
  // If found, retrieve the event
  auto [it, _] = _thread_events.try_emplace(tid, EventPool::instance().get());

  // If the event has been used on a previous I/O task on the same thread, overwrite previous
  // captured state
  it->second.record(stream);
}

void IoContext::sync_all_events()
{
  KVIKIO_NVTX_FUNC_RANGE();
  // No lock needed. All I/O tasks are done, no concurrent access.

  for (auto& [_, event] : _thread_events) {
    event.synchronize();
  }

  // Clear the map. Event destructors return events to the pool
  _thread_events.clear();
}

}  // namespace kvikio::detail
