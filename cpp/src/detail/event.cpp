/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <kvikio/detail/event.hpp>
#include <kvikio/detail/nvtx.hpp>
#include <kvikio/error.hpp>
#include <kvikio/shim/cuda.hpp>

namespace kvikio::detail {

EventPool::Event::Event(CUevent event, CUcontext context) noexcept
  : _event(event), _context(context)
{
}

EventPool::Event::~Event() noexcept
{
  if (_event != nullptr) { EventPool::instance().put(_event, _context); }
}

EventPool::Event::Event(Event&& other) noexcept
  : _event(std::exchange(other._event, nullptr)), _context(std::exchange(other._context, nullptr))
{
}

EventPool::Event& EventPool::Event::operator=(Event&& other) noexcept
{
  if (this != &other) {
    if (_event != nullptr) { EventPool::instance().put(_event, _context); }
    _event   = std::exchange(other._event, nullptr);
    _context = std::exchange(other._context, nullptr);
  }
  return *this;
}

CUevent EventPool::Event::get() const noexcept { return _event; }

CUcontext EventPool::Event::context() const noexcept { return _context; }

void EventPool::Event::record(CUstream stream)
{
  CUDA_DRIVER_TRY(cudaAPI::instance().EventRecord(_event, stream));
}

void EventPool::Event::synchronize()
{
  KVIKIO_NVTX_FUNC_RANGE();
  CUDA_DRIVER_TRY(cudaAPI::instance().EventSynchronize(_event));
}

EventPool::Event EventPool::get()
{
  KVIKIO_NVTX_FUNC_RANGE();
  CUcontext ctx{};
  CUDA_DRIVER_TRY(cudaAPI::instance().CtxGetCurrent(&ctx));
  KVIKIO_EXPECT(ctx != nullptr, "No CUDA context is current");
  return get(ctx);
}

EventPool::Event EventPool::get(CUcontext context)
{
  KVIKIO_EXPECT(context != nullptr, "No CUDA context is current");
  CUevent event{};
  {
    std::lock_guard const lock(_mutex);
    auto it = _pools.find(context);
    if (it != _pools.end() && !it->second.empty()) {
      event = it->second.back();
      it->second.pop_back();
    }
  }

  // Create the event outside the lock
  if (event == nullptr) {
    CUDA_DRIVER_TRY(cudaAPI::instance().EventCreate(&event, CU_EVENT_DISABLE_TIMING));
  }

  return Event(event, context);
}

void EventPool::put(CUevent event, CUcontext context) noexcept
{
  KVIKIO_NVTX_FUNC_RANGE();
  if (event == nullptr) { return; }

  try {
    std::lock_guard const lock(_mutex);
    _pools[context].push_back(event);
  } catch (...) {
    // If returning to pool fails, destroy the event
    cudaAPI::instance().EventDestroy(event);
  }
}

std::size_t EventPool::num_free_events(CUcontext context) const
{
  std::lock_guard const lock(_mutex);
  auto it = _pools.find(context);
  return (it != _pools.end()) ? it->second.size() : 0;
}

std::size_t EventPool::total_free_events() const
{
  std::lock_guard const lock(_mutex);
  std::size_t total{0};
  for (auto const& [ctx, events] : _pools) {
    total += events.size();
  }
  return total;
}

EventPool& EventPool::instance()
{
  static EventPool pool;
  return pool;
}

}  // namespace kvikio::detail
