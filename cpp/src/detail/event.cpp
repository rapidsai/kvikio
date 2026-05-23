/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <exception>
#include <utility>

#include <kvikio/detail/event.hpp>
#include <kvikio/detail/nvtx.hpp>
#include <kvikio/error.hpp>
#include <kvikio/logger.hpp>
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
    if (_event != nullptr) {
      // Return this event to the pool
      _pool->put(_event, _cuda_context);
    }
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
  KVIKIO_CUDA_DRIVER_TRY(cudaAPI::instance().EventRecord(_event, stream));
}

void EventPool::Event::synchronize()
{
  KVIKIO_NVTX_FUNC_RANGE();
  KVIKIO_CUDA_DRIVER_TRY(cudaAPI::instance().EventSynchronize(_event));
}

bool EventPool::Event::is_done() const
{
  auto const status = cudaAPI::instance().EventQuery(_event);
  if (status == CUDA_SUCCESS) { return true; }
  if (status == CUDA_ERROR_NOT_READY) { return false; }
  // Any other return code is an error.
  KVIKIO_CUDA_DRIVER_TRY(status);
  // Unreachable. Macro throws on non-success codes.
  return false;
}

EventPool::Event EventPool::get()
{
  KVIKIO_NVTX_FUNC_RANGE();
  CUcontext ctx{};
  KVIKIO_CUDA_DRIVER_TRY(cudaAPI::instance().CtxGetCurrent(&ctx));
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
    // Create an event outside the lock to improve performance. The pool is not updated here. The
    // returned Event object will automatically return the event to the pool when it goes out of
    // scope
    KVIKIO_CUDA_DRIVER_TRY(cudaAPI::instance().EventCreate(&event, CU_EVENT_DISABLE_TIMING));
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
    // push_back can throw on allocator failure (e.g., out-of-memory). The event cannot stay
    // cached, so destroy it to release its CUDA resources.
    KVIKIO_LOG_ERROR(e.what());
    try {
      KVIKIO_CUDA_DRIVER_TRY(cudaAPI::instance().EventDestroy(event));
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
