/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cstddef>
#include <exception>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>

#include <curl/curl.h>

#include <kvikio/bounce_buffer.hpp>
#include <kvikio/defaults.hpp>
#include <kvikio/detail/bounce_buffer_cache.hpp>
#include <kvikio/detail/multi_poll_reactor.hpp>
#include <kvikio/detail/stream.hpp>
#include <kvikio/error.hpp>
#include <kvikio/remote_handle.hpp>
#include <kvikio/shim/cuda.hpp>
#include <kvikio/shim/libcurl.hpp>
#include <kvikio/utils.hpp>

namespace kvikio::detail {

RemoteMultiAggregateContext::RemoteMultiAggregateContext(std::size_t num_subranges)
  : _subranges_left{num_subranges}
{
  KVIKIO_EXPECT(num_subranges > 0,
                "RemoteMultiAggregateContext requires at least one sub-range",
                std::invalid_argument);
}

void RemoteMultiAggregateContext::on_subrange_complete(std::size_t bytes)
{
  _total_bytes.fetch_add(bytes, std::memory_order_relaxed);
  // Last thread to decrement to zero fulfills the promise.
  // _subranges_left needs "release" in order to publish a thread's own relaxed _total_bytes. It
  // also needs "acquire" in order to load other threads' relaxed _total_bytes to fulfill the
  // _promise. No special handling is needed for _first_exception, because it is updated under a
  // mutex, which provides the memory-ordering guarantee.
  if (_subranges_left.fetch_sub(1, std::memory_order_acq_rel) == 1) {
    std::lock_guard<std::mutex> const lock(_exception_mutex);
    if (_first_exception) {
      _promise.set_exception(_first_exception);
    } else {
      _promise.set_value(_total_bytes.load(std::memory_order_relaxed));
    }
  }
}

void RemoteMultiAggregateContext::on_subrange_failed(std::exception_ptr eptr)
{
  {
    std::lock_guard<std::mutex> const lock(_exception_mutex);
    if (!_first_exception) { _first_exception = eptr; }
  }
  // Last thread to decrement to zero fulfills the promise.
  if (_subranges_left.fetch_sub(1, std::memory_order_acq_rel) == 1) {
    std::lock_guard<std::mutex> const lock(_exception_mutex);
    _promise.set_exception(_first_exception);
  }
}

std::future<std::size_t> RemoteMultiAggregateContext::get_future() { return _promise.get_future(); }

MultiPollReactor::MultiPollReactor(MultiReactorPool* pool) : _pool{pool}
{
  KVIKIO_EXPECT(
    _pool != nullptr, "MultiPollReactor requires a non-null pool", std::invalid_argument);
  // Force LibCurl global init before we create the multi handle.
  std::ignore = LibCurl::instance();
  _curl_multi = curl_multi_init();
  KVIKIO_EXPECT(_curl_multi != nullptr, "curl_multi_init() failed", std::runtime_error);
  _io_thread = std::thread(&MultiPollReactor::io_thread_main, this);
}

MultiPollReactor::~MultiPollReactor() noexcept
{
  // Intentionally empty body. `MultiReactorPool` is a leaked-pointer singleton, so its
  // `_reactors` vector and the `std::unique_ptr<MultiPollReactor>` elements inside it
  // are never destroyed. We declare this dtor so the type is complete and usable in
  // `std::unique_ptr`. Running it would call dtor on an unjoined thread and call
  // `std::terminate()`.
}

void MultiPollReactor::wakeup() noexcept { std::ignore = curl_multi_wakeup(_curl_multi); }

void MultiPollReactor::submit(std::unique_ptr<RemoteMultiTransfer> transfer)
{
  std::exception_ptr fail_reason;
  {
    std::lock_guard<std::mutex> const lock(_submit_mutex);
    if (_pool->is_dead()) {
      // The pool is dead. Fail this transfer immediately rather than pushing into an inbox that
      // will never be drained. Reading death_reason inside the lock is overkill but harmless.
      fail_reason = _pool->death_reason();
    } else {
      _inbox.push_back(std::move(transfer));
    }
  }
  if (fail_reason) {
    transfer->aggregate->on_subrange_failed(fail_reason);
    return;
  }
  wakeup();
}

void MultiPollReactor::io_thread_main()
{
  using DeviceCache = BounceBufferCachePerThreadAndContext<CudaPinnedAllocator>;
  try {
    while (!_pool->is_dead()) {
      // Set true if stage (1) defers at least one device transfer because the bounce-buffer cache
      // was at cap. Stage (4) shortens the poll timeout when this is set so the cache is
      // re-checked promptly after recycle callbacks fire.
      bool has_deferred_device_transfer = false;

      // (1) Drain submission queue. Walk the inbox, attaching each ready handle to the multi
      // handle. For device transfers, check out a pinned bounce buffer from the cache first; if
      // the cache is at cap, leave the transfer in the inbox for a later iteration.
      {
        std::lock_guard<std::mutex> const lock(_submit_mutex);
        for (auto it = _inbox.begin(); it != _inbox.end();) {
          auto& transfer = *it;

          if (transfer->is_device) {
            std::optional<CudaPinnedBounceBufferPool::Buffer> maybe;
            {
              PushAndPopContext c(transfer->device_ctx);
              maybe = DeviceCache::instance().try_get(transfer->device_ctx);
            }
            if (!maybe) {
              has_deferred_device_transfer = true;
              ++it;
              continue;
            }
            transfer->buffer            = std::move(*maybe);
            transfer->ctx.pinned_buffer = transfer->buffer.get();
          }

          CURL* easy    = transfer->curl->handle();
          auto const mc = curl_multi_add_handle(_curl_multi, easy);
          if (mc != CURLM_OK) {
            // This transfer is now in transit between _inbox and _in_flight. fail_all_pending (run
            // on the catch path below) iterates only those two containers. We must fail this
            // transfer's aggregate here to maintain the per-aggregate sub-range count invariant.
            // Also recycle the just-checked-out device buffer if any.
            if (transfer->is_device && transfer->buffer.get() != nullptr) {
              PushAndPopContext c(transfer->device_ctx);
              DeviceCache::instance().recycle_now(transfer->device_ctx,
                                                  std::move(transfer->buffer));
            }
            transfer->aggregate->on_subrange_failed(std::make_exception_ptr(std::runtime_error(
              std::string("curl_multi_add_handle: ") + curl_multi_strerror(mc))));
            _inbox.erase(it);
            KVIKIO_FAIL(std::string("curl_multi_add_handle: ") + curl_multi_strerror(mc),
                        std::runtime_error);
          }
          _in_flight.emplace(easy, std::move(transfer));
          it = _inbox.erase(it);
        }
      }

      // (2) Drive transfers in a non-blocking way.
      int running_handles   = 0;
      auto const perform_mc = curl_multi_perform(_curl_multi, &running_handles);
      KVIKIO_EXPECT(perform_mc == CURLM_OK,
                    std::string("curl_multi_perform: ") + curl_multi_strerror(perform_mc),
                    std::runtime_error);

      // (3) Drain completions.
      int msgs_left = 0;
      while (auto* msg = curl_multi_info_read(_curl_multi, &msgs_left)) {
        if (msg->msg != CURLMSG_DONE) { continue; }
        auto* easy = msg->easy_handle;
        auto res   = msg->data.result;

        auto it = _in_flight.find(easy);
        KVIKIO_EXPECT(it != _in_flight.end(),
                      "MultiPollReactor: completion for unknown handle",
                      std::runtime_error);
        auto transfer = std::move(it->second);
        _in_flight.erase(it);
        // Critical ordering: remove from multi BEFORE the transfer (and its CurlHandle) is
        // destroyed at end of scope. Otherwise libcurl undefined behavior.
        auto const remove_mc = curl_multi_remove_handle(_curl_multi, easy);
        KVIKIO_EXPECT(remove_mc == CURLM_OK,
                      std::string("curl_multi_remove_handle: ") + curl_multi_strerror(remove_mc),
                      std::runtime_error);

        if (res == CURLE_OK && !transfer->ctx.overflow_error) {
          try {
            if (transfer->is_device) {
              // Phase 1 (network -> pinned) done. Now schedule Phase 2 (pinned -> device) on
              // this (thread, ctx) stream and hand the buffer to a cuLaunchHostFunc recycle
              // callback so the cache slot is returned when the H2D drains.
              PushAndPopContext c(transfer->device_ctx);
              CUstream stream = StreamCachePerThreadAndContext::get();
              KVIKIO_CUDA_DRIVER_TRY(
                cudaAPI::instance().MemcpyHtoDAsync(convert_void2deviceptr(transfer->device_dst),
                                                    transfer->buffer.get(),
                                                    transfer->ctx.size,
                                                    stream));
              transfer->aggregate->io_event_barrier->record_event(stream);
              DeviceCache::instance().recycle_after(
                transfer->device_ctx, std::move(transfer->buffer), stream);
            }
            transfer->aggregate->on_subrange_complete(transfer->ctx.size);
          } catch (...) {
            // Stage (3) CUDA/H2D path failed. Recycle the buffer if it is still held (i.e.
            // recycle_after did not run) and fail the aggregate so the caller's future is
            // resolved instead of left broken at scope exit.
            if (transfer->is_device && transfer->buffer.get() != nullptr) {
              try {
                PushAndPopContext c(transfer->device_ctx);
                DeviceCache::instance().recycle_now(transfer->device_ctx,
                                                    std::move(transfer->buffer));
              } catch (...) {
                // Best-effort recycle; if the context push fails the buffer leaks.
              }
            }
            transfer->aggregate->on_subrange_failed(std::current_exception());
          }
        } else {
          std::stringstream ss;
          ss << "curl_multi transfer failed (" << curl_easy_strerror(res) << ")";
          if (transfer->ctx.overflow_error) {
            ss << " [server returned more bytes than requested; maybe range support "
                  "missing?]";
          }
          // No H2D was submitted on the failure path. Recycle the buffer immediately.
          if (transfer->is_device && transfer->buffer.get() != nullptr) {
            PushAndPopContext c(transfer->device_ctx);
            DeviceCache::instance().recycle_now(transfer->device_ctx, std::move(transfer->buffer));
          }
          transfer->aggregate->on_subrange_failed(
            std::make_exception_ptr(std::runtime_error(ss.str())));
        }
        // transfer (unique_ptr) drops here, returning easy to the LibCurl pool.
      }

      // (4) Wait for activity, wakeup, or a bounded timeout. Shorten the timeout when at least
      // one device transfer is deferred on the cache so the cache is re-checked promptly after
      // recycle callbacks fire (otherwise we could sleep up to 1 s for a callback-driven event
      // that does not raise libcurl socket activity).
      int const poll_timeout_ms = has_deferred_device_transfer ? 10 : 1000;
      auto const poll_mc        = curl_multi_poll(_curl_multi,
                                           nullptr,  // extra_fds
                                           0,        // extra_nfds
                                           poll_timeout_ms,
                                           nullptr);  // numfds
      KVIKIO_EXPECT(poll_mc == CURLM_OK,
                    std::string("curl_multi_poll: ") + curl_multi_strerror(poll_mc),
                    std::runtime_error);
    }
  } catch (...) {
    // Any libcurl multi-API error caught above declares pool-wide death. The first reactor to
    // signal wins. Subsequent signals are silently ignored.
    _pool->signal_death(std::current_exception());
  }
  // At this point, we have caught the exception above, or noticed that _pool->is_dead() at loop
  // top. In either case, now drain our own state with the recorded reason to satisfy the
  // aggregate's _promise, so that no call's future.get() hangs.
  fail_all_pending(_pool->death_reason());
}

void MultiPollReactor::fail_all_pending(std::exception_ptr eptr)
{
  using DeviceCache = BounceBufferCachePerThreadAndContext<CudaPinnedAllocator>;

  // Drain the inbox under the submit mutex. New submissions are blocked from accumulating by the
  // _pool->is_dead() check in submit(), which is already true by the time we get here.
  {
    std::lock_guard<std::mutex> const lock(_submit_mutex);
    while (!_inbox.empty()) {
      auto transfer = std::move(_inbox.front());
      _inbox.pop_front();
      // Inbox transfers that already had a buffer checked out (stage (1) succeeded but stage (2)
      // never ran) must return that buffer to the cache. A null `buffer.get()` covers both host
      // transfers and not-yet-checked-out device transfers.
      if (transfer->is_device && transfer->buffer.get() != nullptr) {
        PushAndPopContext c(transfer->device_ctx);
        DeviceCache::instance().recycle_now(transfer->device_ctx, std::move(transfer->buffer));
      }
      transfer->aggregate->on_subrange_failed(eptr);
    }
  }

  // In-flight is touched only by the I/O thread, which is us, so no lock needed.
  for (auto& [easy, transfer] : _in_flight) {
    // Best-effort removal.
    // TODO: Known issue: if curl_multi_remove_handle fails (rarely happens), the easy handle
    // remains attached to the multi handle. _in_flight.clear() below then destroys the transfer's
    // CurlHandle, which unconditionally returns the easy handle to the LibCurl free pool. A future
    // caller that pulls this handle would operate on a handle that libcurl still considers
    // attached, which is undefined behavior.
    std::ignore = curl_multi_remove_handle(_curl_multi, easy);
    // In-flight device transfers had their buffer checked out by stage (1) but no H2D submitted
    // yet (stage (3) is where the H2D goes, and we never reached it for these). Return the buffer
    // to the cache.
    if (transfer->is_device && transfer->buffer.get() != nullptr) {
      PushAndPopContext c(transfer->device_ctx);
      DeviceCache::instance().recycle_now(transfer->device_ctx, std::move(transfer->buffer));
    }
    transfer->aggregate->on_subrange_failed(eptr);
  }
  _in_flight.clear();
}

MultiReactorPool::MultiReactorPool() : _dispatch{defaults::remote_io_reactor_dispatch()}
{
  // Force LibCurl global init before any reactor opens a multi handle.
  std::ignore = LibCurl::instance();

  auto const n = defaults::remote_io_num_reactors();
  KVIKIO_EXPECT(n > 0, "remote_io_num_reactors must be a positive integer", std::invalid_argument);
  _reactors.reserve(n);
  for (unsigned int i = 0; i < n; ++i) {
    _reactors.emplace_back(std::make_unique<MultiPollReactor>(this));
  }
}

MultiReactorPool::~MultiReactorPool() noexcept
{
  // Intentionally empty body. The pool is a leaked-pointer singleton so this destructor is never
  // invoked.
}

MultiReactorPool& MultiReactorPool::instance()
{
  // Heap-leaked singleton. The pool, its reactors, and their `std::thread`s are never destroyed.
  // Resources are cleaned on process exit.
  static MultiReactorPool* inst = new MultiReactorPool();
  return *inst;
}

void MultiReactorPool::submit_pread(std::vector<std::unique_ptr<RemoteMultiTransfer>> transfers)
{
  auto const reactor_count = _reactors.size();

  // PER_PREAD: one reactor for the whole pread() call. Preserves per-CURLM connection-pool reuse.
  if (_dispatch == RemoteReactorDispatch::PER_PREAD) {
    auto const idx = _next_reactor_counter.fetch_add(1, std::memory_order_relaxed) % reactor_count;
    for (auto& transfer : transfers) {
      _reactors[idx]->submit(std::move(transfer));
    }
    return;
  }

  // PER_CHUNK: round-robin sub-ranges across reactors.
  for (auto& transfer : transfers) {
    auto const idx = _next_reactor_counter.fetch_add(1, std::memory_order_relaxed) % reactor_count;
    _reactors[idx]->submit(std::move(transfer));
  }
}

bool MultiReactorPool::is_dead() const noexcept
{
  // This function is on a hot path, so we use atomic instead of a mutex.
  return _dead.load(std::memory_order_acquire);
}

std::exception_ptr MultiReactorPool::death_reason() const noexcept
{
  std::lock_guard<std::mutex> const lock(_death_mutex);
  return _death_reason;
}

void MultiReactorPool::signal_death(std::exception_ptr eptr) noexcept
{
  // - The lock is needed to avoid multiple threads updating _death_reason at the same time.
  // - The store needs to stay inside the scope of lock. Otherwise, multiple threads may own the
  // mutex at different point of time and the last thread writes to _death_reason, whereas here we
  // want the first thread to win.
  // - The store needs to use `release` to pair with the load's `acquire` in `is_dead()`.
  // - The load can be relaxed. `acquire` or `seq_cst` will be an overkill.
  {
    std::lock_guard<std::mutex> const lock(_death_mutex);
    // Only the first reactor I/O thread that reaches here updates _death_reason and performs the
    // wakeup. Subsequent calls will early exit.
    if (_dead.load(std::memory_order_relaxed)) { return; }
    _death_reason = eptr;
    _dead.store(true, std::memory_order_release);
  }

  // Wake up every reactor out of its curl_multi_poll so they notice _dead promptly.
  // At this point the caller's own reactor just exited the loop body to enter the catch. So
  // including it here is harmless.
  for (auto const& r : _reactors) {
    r->wakeup();
  }
}

}  // namespace kvikio::detail
