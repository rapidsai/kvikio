/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
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

MultiPollReactor::MultiPollReactor(MultiReactorPool* pool,
                                   std::optional<std::size_t> max_concurrent_requests)
  : _pool{pool}, _request_limiter{max_concurrent_requests}
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

void MultiPollReactor::submit(std::vector<std::unique_ptr<RemoteMultiTransfer>> transfers)
{
  if (transfers.empty()) { return; }
  std::exception_ptr fail_reason;
  {
    std::lock_guard<std::mutex> const lock(_submit_mutex);
    if (_pool->is_dead()) {
      // The pool is dead. Fail the batch immediately rather than pushing into an inbox that will
      // never be drained. Reading death_reason inside the lock is overkill but harmless.
      fail_reason = _pool->death_reason();
    } else {
      for (auto& transfer : transfers) {
        _inbox.push_back(std::move(transfer));
      }
    }
  }
  if (fail_reason) {
    for (auto& transfer : transfers) {
      transfer->aggregate->on_subrange_failed(fail_reason);
    }
    return;
  }
  wakeup();
}

void MultiPollReactor::io_thread_main()
{
  using BounceBufferCache = BounceBufferCachePerThreadAndContext<CudaPinnedAllocator>;
  try {
    while (!_pool->is_dead()) {
      // (1) Splice newly submitted transfers out of the shared inbox. The mutexed section is
      // pointer moves only, so submitters (by other threads) never wait on admission work (by the
      // current IO reactor thread).
      {
        std::lock_guard<std::mutex> const lock(_submit_mutex);
        if (_pending.empty()) {
          std::swap(_pending, _inbox);
        } else {
          while (!_inbox.empty()) {
            _pending.push_back(std::move(_inbox.front()));
            _inbox.pop_front();
          }
        }
      }

      // Admission walk over the reactor-private pending queue, lock-free. Entries pop off the
      // front and are either admitted to libcurl or moved to `deferred_transfers`; `_pending` is rebuilt at
      // the end. This keeps the walk O(n) with no mid-deque erases and preserves FIFO order across
      // iterations.
      std::deque<std::unique_ptr<RemoteMultiTransfer>> deferred_transfers;
      // Contexts whose bounce-buffer shard has already missed during this walk. Distinct contexts
      // are few (typically one), so a flat vector with linear find beats a hash set.
      std::vector<CUcontext> exhausted_ctxs;
      while (!_pending.empty()) {
        auto transfer = std::move(_pending.front());
        _pending.pop_front();
        try {
          // A ctx that already missed the cache this walk cannot admit further device transfers
          // now. Defer without acquiring a limiter slot or touching CUDA. A recycle callback may
          // free a buffer mid-walk, making this pessimistic by one loop iteration at most.
          if (transfer->is_device &&
              std::find(exhausted_ctxs.begin(), exhausted_ctxs.end(), transfer->device_ctx) !=
                exhausted_ctxs.end()) {
            deferred_transfers.push_back(std::move(transfer));
            continue;
          }

          // Bound this reactor's in-flight requests to its private share of the global budget.
          // Once the limiter is at capacity every remaining entry would fail the same check, so
          // defer them all wholesale in FIFO order.
          auto slot = _request_limiter.try_acquire();
          if (!slot) {
            deferred_transfers.push_back(std::move(transfer));
            while (!_pending.empty()) {
              deferred_transfers.push_back(std::move(_pending.front()));
              _pending.pop_front();
            }
            break;
          }

          if (transfer->is_device) {
            std::optional<CudaPinnedBounceBufferPool::Buffer> maybe;
            {
              PushAndPopContext c(transfer->device_ctx);
              maybe = BounceBufferCache::instance().try_get(transfer->device_ctx);
            }
            if (!maybe) {
              // First miss for this ctx in this walk. `slot` auto-releases at end of scope, so a
              // later host transfer or other-ctx device transfer can still be admitted.
              exhausted_ctxs.push_back(transfer->device_ctx);
              deferred_transfers.push_back(std::move(transfer));
              continue;
            }
            transfer->buffer            = std::move(*maybe);
            transfer->ctx.pinned_buffer = transfer->buffer.get();
          }

          CURL* easy    = transfer->curl->handle();
          auto const mc = curl_multi_add_handle(_curl_multi, easy);
          if (mc != CURLM_OK) {
            // Fail this transfer's aggregate here to maintain the per-aggregate sub-range count
            // invariant, then null the local pointer so the catch below does not requeue an
            // already-notified transfer. Also recycle the just-checked-out device buffer if any.
            if (transfer->is_device && transfer->buffer.get() != nullptr) {
              PushAndPopContext c(transfer->device_ctx);
              BounceBufferCache::instance().recycle_now(transfer->device_ctx,
                                                        std::move(transfer->buffer));
            }
            transfer->aggregate->on_subrange_failed(std::make_exception_ptr(std::runtime_error(
              std::string("curl_multi_add_handle: ") + curl_multi_strerror(mc))));
            transfer.reset();
            KVIKIO_FAIL(std::string("curl_multi_add_handle: ") + curl_multi_strerror(mc),
                        std::runtime_error);
          }
          // The transfer owns its slot from here on. The slot returns to the limiter when the
          // transfer is destroyed.
          transfer->slot = std::move(slot);
          _in_flight.emplace(easy, std::move(transfer));
        } catch (...) {
          // Requeue the in-hand transfer (unless already failed above) and the already-deferred
          // entries, so fail_all_pending, which drains `_pending`, resolves their aggregates.
          if (transfer) { _pending.push_front(std::move(transfer)); }
          while (!deferred_transfers.empty()) {
            _pending.push_front(std::move(deferred_transfers.back()));
            deferred_transfers.pop_back();
          }
          throw;
        }
      }
      // The walk drained `_pending`; the deferred entries become the new pending queue.
      std::swap(_pending, deferred_transfers);

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
              BounceBufferCache::instance().recycle_after(
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
                BounceBufferCache::instance().recycle_now(transfer->device_ctx,
                                                          std::move(transfer->buffer));
              } catch (...) {
                // Best-effort recycle; if the context push fails the buffer leaks.
              }
            }
            transfer->aggregate->on_subrange_failed(std::current_exception());
          }
        } else {
          std::stringstream ss;
          // Prefer the handle's recorded error buffer, which is usually more specific than
          // curl_easy_strerror (for example "The requested URL returned error: 403"). Fall back to
          // the generic strerror text when libcurl recorded no message.
          auto const msg = transfer->curl->error_message();
          ss << "curl_multi transfer failed ("
             << (msg.empty() ? std::string{curl_easy_strerror(res)} : msg) << ")";
          if (transfer->ctx.overflow_error) {
            ss << " [server returned more bytes than requested; maybe range support "
                  "missing?]";
          }
          // No H2D was submitted on the failure path. Recycle the buffer immediately.
          if (transfer->is_device && transfer->buffer.get() != nullptr) {
            PushAndPopContext c(transfer->device_ctx);
            BounceBufferCache::instance().recycle_now(transfer->device_ctx,
                                                      std::move(transfer->buffer));
          }
          transfer->aggregate->on_subrange_failed(
            std::make_exception_ptr(std::runtime_error(ss.str())));
        }
        // transfer (unique_ptr) drops here, returning easy to the LibCurl pool and releasing the
        // transfer's concurrency slot so a deferred request can be admitted.
      }

      // (4) Wait for activity, wakeup, or a bounded timeout. Shorten the timeout while transfers
      // remain deferred in `_pending`, so admission is retried promptly. Recycle callbacks that
      // free a cache slot and completions that free a limiter slot do not necessarily raise
      // libcurl socket activity, so without this we could sleep up to 1 s before retrying a
      // deferred transfer.
      int const poll_timeout_ms = _pending.empty() ? 1000 : 10;
      auto const poll_mc        = curl_multi_poll(_curl_multi,
                                           nullptr,          // extra_fds
                                           0,                // extra_nfds
                                           poll_timeout_ms,  // timeout_ms
                                           nullptr);         // numfds
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
  using BounceBufferCache = BounceBufferCachePerThreadAndContext<CudaPinnedAllocator>;

  // Drain the inbox under the submit mutex. New submissions are blocked from accumulating by the
  // _pool->is_dead() check in submit(), which is already true by the time we get here. Inbox
  // entries have not been through admission, so they hold no bounce buffer and no limiter slot.
  {
    std::lock_guard<std::mutex> const lock(_submit_mutex);
    while (!_inbox.empty()) {
      auto transfer = std::move(_inbox.front());
      _inbox.pop_front();
      transfer->aggregate->on_subrange_failed(eptr);
    }
  }

  // Drain the deferred queue. It is touched only by the I/O thread, which is us, so no lock
  // needed. Entries requeued by the stage (1) exception path may carry a checked-out bounce
  // buffer; return it to the cache. A null `buffer.get()` covers host transfers and
  // not-yet-checked-out device transfers.
  while (!_pending.empty()) {
    auto transfer = std::move(_pending.front());
    _pending.pop_front();
    if (transfer->is_device && transfer->buffer.get() != nullptr) {
      PushAndPopContext c(transfer->device_ctx);
      BounceBufferCache::instance().recycle_now(transfer->device_ctx, std::move(transfer->buffer));
    }
    transfer->aggregate->on_subrange_failed(eptr);
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
      BounceBufferCache::instance().recycle_now(transfer->device_ctx, std::move(transfer->buffer));
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

  auto const max_total = defaults::remote_io_max_concurrent_requests();
  std::optional<std::size_t> const per_reactor_max =
    (max_total == 0) ? std::nullopt : std::optional{std::max<std::size_t>(max_total / n, 1)};

  _reactors.reserve(n);
  for (unsigned int i = 0; i < n; ++i) {
    _reactors.emplace_back(std::make_unique<MultiPollReactor>(this, per_reactor_max));
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
  // The whole batch is submitted in one call: one lock acquisition and one wakeup.
  if (_dispatch == RemoteReactorDispatch::PER_PREAD) {
    auto const idx = _next_reactor_counter.fetch_add(1, std::memory_order_relaxed) % reactor_count;
    _reactors[idx]->submit(std::move(transfers));
    return;
  }

  // PER_CHUNK: round-robin sub-ranges across reactors, but submit per-reactor batches so each
  // reactor pays one lock acquisition and one wakeup per pread instead of one per sub-range.
  std::vector<std::vector<std::unique_ptr<RemoteMultiTransfer>>> buckets(reactor_count);
  for (auto& transfer : transfers) {
    auto const idx = _next_reactor_counter.fetch_add(1, std::memory_order_relaxed) % reactor_count;
    buckets[idx].push_back(std::move(transfer));
  }
  for (std::size_t i = 0; i < reactor_count; ++i) {
    if (!buckets[i].empty()) { _reactors[i]->submit(std::move(buckets[i])); }
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
