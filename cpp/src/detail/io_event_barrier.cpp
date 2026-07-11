/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <mutex>
#include <thread>

#include <kvikio/detail/event.hpp>
#include <kvikio/detail/io_event_barrier.hpp>

namespace kvikio::detail {

IoEventBarrier::IoEventBarrier(CUcontext cuda_context) noexcept : _cuda_context{cuda_context} {}

CUcontext IoEventBarrier::cuda_context() const noexcept { return _cuda_context; }

void IoEventBarrier::record_event(CUstream stream)
{
  CudaEventPool::CudaEvent* event_ptr{nullptr};
  {
    std::lock_guard const lock(_mutex);
    auto const tid = std::this_thread::get_id();
    auto it        = _thread_events.find(tid);
    if (it == _thread_events.end()) {
      it = _thread_events.emplace(tid, CudaEventPool::instance().get()).first;
    }
    event_ptr = &it->second;
  }
  // Note that for the node-based unordered_map, pointers (or references) to either key or data
  // stored in the container can never be invalidated by insertion, even when the corresponding
  // iterator is invalidated. So it is safe to move this function outside the mutex.
  event_ptr->record(stream);
}

void IoEventBarrier::sync_all_events()
{
  std::lock_guard const lock(_mutex);
  for (auto& [tid, event] : _thread_events) {
    event.synchronize();
  }
}

}  // namespace kvikio::detail
