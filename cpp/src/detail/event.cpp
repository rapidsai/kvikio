/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <kvikio/detail/event.hpp>
#include <kvikio/detail/nvtx.hpp>
#include <kvikio/error.hpp>
#include <kvikio/shim/cuda.hpp>

namespace kvikio::detail {

EventPool::Event::Event(EventPool* pool, CUevent event, CUcontext cuda_context) noexcept
  : _pool(pool), _event(event), _cuda_context(cuda_context)
{
}

EventPool::Event::~Event() noexcept
{
  if (_event != nullptr) { _pool->put(_event, _cuda_context); }
}

EventPool::Event::Event(Event&& o) noexcept
  : _pool(std::exchange(o._pool, nullptr)),
    _event(std::exchange(o._event, nullptr)),
    _cuda_context(std::exchange(o._cuda_context, nullptr))
{
}

EventPool::Event& EventPool::Event::operator=(Event&& o) noexcept
{
  if (this != &o) {
    if (_event != nullptr) { _pool->put(_event, _cuda_context); }
    _pool         = std::exchange(o._pool, nullptr);
    _event        = std::exchange(o._event, nullptr);
    _cuda_context = std::exchange(o._cuda_context, nullptr);
  }
  return *this;
}

CUevent EventPool::Event::get() const noexcept { return _event; }

CUcontext EventPool::Event::cuda_context() const noexcept { return _cuda_context; }

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

  CUevent event{};
  {
    std::lock_guard const lock(_mutex);
    // If the key (`ctx`) is found from the pool, assign the search result to `event`
    if (auto it = _pools.find(ctx); it != _pools.end() && !it->second.empty()) {
      event = it->second.back();
      it->second.pop_back();
    }
  }

  if (event == nullptr) {
    // Create an event outside the lock to improve performance.
    // The pool is not updated here; the returned Event object will automatically return the event
    // to the pool when it goes out of scope
    CUDA_DRIVER_TRY(cudaAPI::instance().EventCreate(&event, CU_EVENT_DISABLE_TIMING));
  }

  return Event(this, event, ctx);
}

void EventPool::put(CUevent event, CUcontext cuda_context) noexcept
{
  KVIKIO_NVTX_FUNC_RANGE();
  if (event == nullptr) { return; }

  try {
    std::lock_guard const lock(_mutex);
    _pools[cuda_context].push_back(event);
  } catch (std::exception const& e) {
    KVIKIO_LOG_ERROR(e.what());
    try {
      // If returning to pool fails, destroy the event
      CUDA_DRIVER_TRY(cudaAPI::instance().EventDestroy(event));
    } catch (std::exception const& e) {
      KVIKIO_LOG_ERROR(e.what());
    }
  }
}

std::size_t EventPool::num_free_events(CUcontext cuda_context) const
{
  std::lock_guard const lock(_mutex);
  auto it = _pools.find(cuda_context);
  return (it != _pools.end()) ? it->second.size() : 0;
}

std::size_t EventPool::total_free_events() const
{
  std::lock_guard const lock(_mutex);
  std::size_t total{0};
  for (auto const& [_, events] : _pools) {
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
