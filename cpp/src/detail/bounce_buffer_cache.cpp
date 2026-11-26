/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <bit>
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
namespace {

/// The smallest buffer a sized request gets: one transparent huge page, so that every buffer can be
/// backed by huge pages (see `CudaPinnedAllocator`).
constexpr std::size_t min_sized_buffer = 2 * 1024 * 1024;

/**
 * @brief Size of the buffer that serves a request
 *
 * @param size The number of bytes requested
 * @param full_size The bounce buffer size
 * @return The next power of two of `size`, at least `min_sized_buffer` and at most `full_size`
 */
std::size_t sized_buffer(std::size_t size, std::size_t full_size)
{
  return std::min(full_size, std::bit_ceil(std::max(size, min_sized_buffer)));
}

/**
 * @brief Whether a free buffer can still be handed out under the current bounce buffer size
 *
 * @param size Size of the free buffer
 * @param full_size The bounce buffer size
 * @return True if `size` is `full_size` or a smaller power of two `sized_buffer` can return
 */
bool is_current_size(std::size_t size, std::size_t full_size)
{
  return size == full_size ||
         (size < full_size && size >= min_sized_buffer && std::has_single_bit(size));
}

}  // namespace

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
  return try_get(ctx, defaults::bounce_buffer_size());
}

template <typename Allocator>
std::optional<typename BounceBufferCachePerThreadAndContext<Allocator>::Buffer>
BounceBufferCachePerThreadAndContext<Allocator>::try_get(CUcontext ctx, std::size_t size)
{
  KVIKIO_NVTX_FUNC_RANGE();
  auto& shard          = get_shard(ctx);
  auto const full_size = defaults::bounce_buffer_size();
  auto const wanted    = sized_buffer(size, full_size);

  // Buffers taken off the free list here are destroyed after the lock is released. Their
  // destructors route through BounceBufferPool::put, which deallocates buffers not of its size.
  std::vector<Buffer> dropped;
  std::optional<Buffer> reused_buffer;
  bool may_allocate{false};
  {
    std::lock_guard const lock(shard.mutex);

    // Drop free buffers whose size is no longer current, e.g. after the bounce buffer size changed.
    for (auto it = shard.free.begin(); it != shard.free.end();) {
      if (is_current_size(it->size(), full_size)) {
        ++it;
      } else {
        dropped.push_back(std::move(*it));
        it = shard.free.erase(it);
      }
    }

    // Reuse the smallest free buffer that is large enough, the most recently returned on a tie.
    auto best = shard.free.rend();
    for (auto it = shard.free.rbegin(); it != shard.free.rend(); ++it) {
      if (it->size() >= wanted && (best == shard.free.rend() || it->size() < best->size())) {
        best = it;
      }
    }

    if (best != shard.free.rend()) {
      std::iter_swap(best, shard.free.rbegin());
      reused_buffer = std::move(shard.free.back());
      shard.free.pop_back();
      ++shard.checked_out;
    } else if (!_cap.has_value() ||
               shard.free.size() + shard.checked_out + shard.in_flight < _cap.value()) {
      ++shard.checked_out;
      may_allocate = true;
    } else if (!shard.free.empty()) {
      // At the cap with only smaller buffers free: give one up to make room for a larger one.
      dropped.push_back(std::move(shard.free.back()));
      shard.free.pop_back();
      ++shard.checked_out;
      may_allocate = true;
    }
  }

  dropped.clear();

  if (reused_buffer.has_value() || !may_allocate) { return reused_buffer; }

  try {
    if (wanted == full_size) { return BounceBufferPool<Allocator>::instance().get(); }
    return Buffer(&BounceBufferPool<Allocator>::instance(), Allocator{}.allocate(wanted), wanted);
  } catch (...) {
    std::lock_guard const lock(shard.mutex);
    --shard.checked_out;
    throw;
  }
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
void BounceBufferCachePerThreadAndContext<Allocator>::recycle_after(
  CUcontext ctx, Buffer&& buf, CUstream stream, std::function<void()> on_recycle)
{
  KVIKIO_NVTX_FUNC_RANGE();
  auto& shard = get_shard(ctx);
  auto data = std::make_unique<RecycleCallbackData>(&shard, std::move(buf), std::move(on_recycle));

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
    {
      std::lock_guard const lock(data->shard->mutex);
      --data->shard->in_flight;
      data->shard->free.push_back(std::move(data->buffer));
    }
    if (data->on_recycle) { data->on_recycle(); }
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
    auto const max_total                 = defaults::remote_io_max_concurrent_requests();
    auto const n                         = defaults::remote_io_num_reactors();
    constexpr std::size_t headroom_scale = 2;
    std::optional<std::size_t> const per_reactor_max =
      (max_total == 0) ? std::nullopt
                       : std::optional{std::max<std::size_t>(max_total / n, 1) * headroom_scale};
    return new BounceBufferCachePerThreadAndContext(per_reactor_max);
  }();
  return *_instance;
}

// Explicit instantiations
template class BounceBufferCachePerThreadAndContext<PageAlignedAllocator>;
template class BounceBufferCachePerThreadAndContext<CudaPinnedAllocator>;
template class BounceBufferCachePerThreadAndContext<CudaPageAlignedPinnedAllocator>;

}  // namespace kvikio::detail
