// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
// clang-format on

#include <algorithm>
#include <exception>
#include <memory>
#include <mutex>
#include <optional>
#include <tuple>
#include <utility>

#include <kvikio/bounce_buffer.hpp>
#include <kvikio/defaults.hpp>
#include <kvikio/detail/bounce_buffer_cache.hpp>
#include <kvikio/detail/nvtx.hpp>
#include <kvikio/error.hpp>
#include <kvikio/logger.hpp>
#include <kvikio/shim/cuda.hpp>

namespace kvikio::detail {

template <typename Allocator>
BounceBufferCachePerThreadAndContext<Allocator>::Shard::Shard(std::optional<std::size_t> cap)
{
  if (cap.has_value()) { free.reserve(cap.value()); }
}

template <typename Allocator>
BounceBufferCachePerThreadAndContext<Allocator>::BounceBufferCachePerThreadAndContext(
  std::optional<std::size_t> cap)
  : _cap{cap}
{
}

template <typename Allocator>
std::optional<std::size_t> BounceBufferCachePerThreadAndContext<Allocator>::cap() const noexcept
{
  return _cap;
}

template <typename Allocator>
typename BounceBufferCachePerThreadAndContext<Allocator>::Shard&
BounceBufferCachePerThreadAndContext<Allocator>::get_shard(CUcontext ctx)
{
  auto const key = std::pair{std::this_thread::get_id(), ctx};
  std::lock_guard const lock(_map_mutex);
  auto it = _shards.find(key);
  if (it == _shards.end()) { it = _shards.emplace(key, std::make_unique<Shard>(cap())).first; }
  return *it->second;
}

template <typename Allocator>
std::optional<typename BounceBufferCachePerThreadAndContext<Allocator>::Buffer>
BounceBufferCachePerThreadAndContext<Allocator>::try_get(CUcontext ctx)
{
  KVIKIO_NVTX_FUNC_RANGE();
  auto& shard = get_shard(ctx);
  std::lock_guard const lock(shard.mutex);

  // Discard free buffers whose size no longer matches the current bounce_buffer_size. Their
  // destructors route through BounceBufferPool::put, which deallocates wrong-size buffers.
  auto const current_size = defaults::bounce_buffer_size();
  while (!shard.free.empty() && shard.free.back().size() != current_size) {
    shard.free.pop_back();
  }

  if (!shard.free.empty()) {
    auto buf = std::move(shard.free.back());
    shard.free.pop_back();
    ++shard.checked_out;
    return buf;
  }

  // No buffer available on the free list. Allocate if under cap (or if cap is unlimited).
  auto const total = shard.free.size() + shard.checked_out + shard.in_flight;
  if (_cap.has_value() && total >= _cap.value()) { return std::nullopt; }

  auto buf = BounceBufferPool<Allocator>::instance().get();
  ++shard.checked_out;
  return buf;
}

template <typename Allocator>
void BounceBufferCachePerThreadAndContext<Allocator>::recycle_now(CUcontext ctx, Buffer&& buf)
{
  KVIKIO_NVTX_FUNC_RANGE();
  auto& shard = get_shard(ctx);
  std::lock_guard const lock(shard.mutex);
  --shard.checked_out;
  shard.free.push_back(std::move(buf));
}

template <typename Allocator>
void BounceBufferCachePerThreadAndContext<Allocator>::recycle_after(CUcontext ctx,
                                                                    Buffer&& buf,
                                                                    CUstream stream)
{
  KVIKIO_NVTX_FUNC_RANGE();
  auto& shard = get_shard(ctx);
  auto data   = std::make_unique<RecycleCallbackData>(RecycleCallbackData{&shard, std::move(buf)});

  // Phase A (`checked_out`) ends and Phase B (`in_flight`) starts.
  {
    std::lock_guard const lock(shard.mutex);
    --shard.checked_out;
    ++shard.in_flight;
  }

  try {
    KVIKIO_CUDA_DRIVER_TRY(
      cudaAPI::instance().LaunchHostFunc(stream, &recycle_callback, data.get()));
  } catch (...) {
    // LaunchHostFunc throws, and the callback is never enqueued. `data` still owns the payload, so
    // its destructor returns the buffer to BounceBufferPool during unwinding. The buffer leaves the
    // shard, so we only decrement in_flight and not restore checked_out.
    std::lock_guard const lock(shard.mutex);
    --shard.in_flight;
    throw;
  }

  // The callback owns the heap payload. Here we disown it so this unique_ptr's destructor does
  // not also delete it. If the callback has already run on another thread and freed the payload,
  // `release()` returns a dangling pointer, which we ignore, so that is harmless.
  std::ignore = data.release();
}

template <typename Allocator>
void CUDA_CB BounceBufferCachePerThreadAndContext<Allocator>::recycle_callback(void* user_data)
{
  // Runs on a CUDA driver controlled thread. Must not make CUDA API calls. Must be short.
  std::unique_ptr<RecycleCallbackData> data(static_cast<RecycleCallbackData*>(user_data));
  try {
    std::lock_guard const lock(data->shard->mutex);
    --data->shard->in_flight;
    data->shard->free.push_back(std::move(data->buffer));
  } catch (std::exception const& e) {
    KVIKIO_LOG_ERROR(std::string("BounceBufferCachePerThreadAndContext::recycle_callback: ") +
                     e.what());
  } catch (...) {
    KVIKIO_LOG_ERROR("BounceBufferCachePerThreadAndContext::recycle_callback: unknown exception");
  }
}

template <typename Allocator>
BounceBufferCachePerThreadAndContext<Allocator>&
BounceBufferCachePerThreadAndContext<Allocator>::instance()
{
  KVIKIO_NVTX_FUNC_RANGE();
  static auto* _instance = []() {
    auto const max_total = defaults::remote_io_max_concurrent_requests();
    auto const n         = defaults::remote_io_num_reactors();
    std::optional<std::size_t> const per_reactor_max =
      (max_total == 0) ? std::nullopt : std::optional{std::max<std::size_t>(max_total / n, 1)};
    return new BounceBufferCachePerThreadAndContext(per_reactor_max);
  }();
  return *_instance;
}

// Explicit instantiations
template class BounceBufferCachePerThreadAndContext<PageAlignedAllocator>;
template class BounceBufferCachePerThreadAndContext<CudaPinnedAllocator>;
template class BounceBufferCachePerThreadAndContext<CudaPageAlignedPinnedAllocator>;

}  // namespace kvikio::detail
