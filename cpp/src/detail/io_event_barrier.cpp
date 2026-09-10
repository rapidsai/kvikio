/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <mutex>
#include <thread>
#include <utility>
#include <vector>

#include <kvikio/detail/event.hpp>
#include <kvikio/detail/io_event_barrier.hpp>
#include <kvikio/error.hpp>

namespace kvikio::detail {

IoEventBarrier::IoEventBarrier(CUcontext cuda_context) noexcept : _cuda_context{cuda_context} {}

CUcontext IoEventBarrier::cuda_context() const noexcept { return _cuda_context; }

void IoEventBarrier::record_event(CUstream stream)
{
  CudaEventPool::CudaEvent* event_ptr{nullptr};
  auto const tid = std::this_thread::get_id();
  {
    std::lock_guard const lock(_mutex);
    if (auto it = _thread_events.find(tid); it != _thread_events.end()) { event_ptr = &it->second; }
  }

  if (event_ptr == nullptr) {
    auto event = CudaEventPool::instance().get();
    {
      std::lock_guard const lock(_mutex);
      auto [it, inserted] = _thread_events.emplace(tid, std::move(event));
      KVIKIO_EXPECT(inserted, "New event insertion failed unexpectedly.");
      event_ptr = &it->second;
    }
  }

  // Note that for the node-based unordered_map, pointers (or references) to either key or data
  // stored in the container can never be invalidated by insertion, even when the corresponding
  // iterator is invalidated. So it is safe to move this function outside the mutex.
  event_ptr->record(stream);
}

void IoEventBarrier::sync_all_events()
{
  std::vector<CudaEventPool::CudaEvent*> events;
  {
    std::lock_guard const lock(_mutex);
    events.reserve(_thread_events.size());
    for (auto& [tid, event] : _thread_events) {
      events.push_back(&event);
    }
  }
  for (auto* event : events) {
    event->synchronize();
  }
}

}  // namespace kvikio::detail
