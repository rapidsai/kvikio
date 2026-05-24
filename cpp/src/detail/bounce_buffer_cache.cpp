/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <exception>
#include <memory>
#include <mutex>
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
BounceBufferCachePerThreadAndContext<Allocator>::BounceBufferCachePerThreadAndContext(
  std::size_t cap)
  : _cap{cap}
{
}

template <typename Allocator>
std::size_t BounceBufferCachePerThreadAndContext<Allocator>::cap() const noexcept
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
  if (it == _shards.end()) { it = _shards.emplace(key, std::make_unique<Shard>()).first; }
  return *it->second;
}

template <typename Allocator>
std::optional<typename BounceBufferCachePerThreadAndContext<Allocator>::Buffer>
BounceBufferCachePerThreadAndContext<Allocator>::try_get(CUcontext ctx)
{
  KVIKIO_NVTX_FUNC_RANGE();
  auto& shard = get_shard(ctx);

  std::lock_guard const lock(shard.mutex);

  // Discard any free buffers whose size does not match the current bounce_buffer_size. Their
  // destructors route through `BounceBufferPool::put`, which deallocates wrong-size buffers.
  // Mirrors `BounceBufferPool::_ensure_buffer_size` for free-list-held buffers in the cache.
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
  if (_cap != 0 && total >= _cap) { return std::nullopt; }

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

  KVIKIO_CUDA_DRIVER_TRY(cudaAPI::instance().LaunchHostFunc(stream, &recycle_callback, data.get()));
  // Ownership of the heap payload transfers to the callback.
  // It is okay if at this point, the callback has already be executed, and the heap payload
  // destroyed, in which case, `release` here would simply return nullptr instead of a pointer to
  // the managed object.
  std::ignore = data.release();
}

template <typename Allocator>
void CUDA_CB BounceBufferCachePerThreadAndContext<Allocator>::recycle_callback(void* user_data)
{
  // Runs on a CUDA driver controlled thread. Must not make CUDA API calls. Must be short.
  std::unique_ptr<RecycleCallbackData> data(static_cast<RecycleCallbackData*>(user_data));
  try {
    std::lock_guard const lock(data->shard->mutex);
    data->shard->free.push_back(std::move(data->buffer));
    --data->shard->in_flight;
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
  static auto* _instance =
    new BounceBufferCachePerThreadAndContext(defaults::num_bounce_buffers_per_cache());
  return *_instance;
}

// Explicit instantiations
template class BounceBufferCachePerThreadAndContext<PageAlignedAllocator>;
template class BounceBufferCachePerThreadAndContext<CudaPinnedAllocator>;
template class BounceBufferCachePerThreadAndContext<CudaPageAlignedPinnedAllocator>;

}  // namespace kvikio::detail
