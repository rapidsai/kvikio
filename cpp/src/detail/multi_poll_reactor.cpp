/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <cstddef>
#include <exception>
#include <memory>
#include <optional>
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

CurlMultiAttachment::CurlMultiAttachment(CURLM* multi, CURL* easy) noexcept
  : _multi{multi}, _easy{easy}
{
}

CurlMultiAttachment::~CurlMultiAttachment()
{
  // Best-effort detach on the reactor I/O thread (CURLM is not thread-safe). If
  // curl_multi_remove_handle fails (rare), the easy handle stays attached and the owning CurlHandle
  // still returns it to the LibCurl pool, which libcurl treats as undefined behavior. A destructor
  // has no better recovery; this matches the prior best-effort removal in fail_all_pending.
  if (_multi != nullptr && _easy != nullptr) {
    std::ignore = curl_multi_remove_handle(_multi, _easy);
  }
}

CurlMultiAttachment::CurlMultiAttachment(CurlMultiAttachment&& o) noexcept
  : _multi{std::exchange(o._multi, nullptr)}, _easy{std::exchange(o._easy, nullptr)}
{
}

CurlMultiAttachment& CurlMultiAttachment::operator=(CurlMultiAttachment&& o) noexcept
{
  if (this != &o) {
    // Detach whatever this guard currently holds before taking over o's handle.
    if (_multi != nullptr && _easy != nullptr) {
      std::ignore = curl_multi_remove_handle(_multi, _easy);
    }
    _multi = std::exchange(o._multi, nullptr);
    _easy  = std::exchange(o._easy, nullptr);
  }
  return *this;
}

RemoteMultiTransfer::~RemoteMultiTransfer()
{
  using BounceBufferCache = BounceBufferCachePerThreadAndContext<CudaPinnedAllocator>;
  // A device transfer that still owns its bounce buffer reaches here only on a failure path; the
  // success path moves the buffer into recycle_after, leaving buffer.get() == nullptr.
  //
  // Thread-affinity invariant: the cache is sharded by (this_thread::get_id(), ctx), so a transfer
  // holding a buffer MUST be destroyed on the same reactor I/O thread that checked it out in stage
  // (1). All buffer-holding destructions (in-flight drain, admission-walk reset(), completion drop)
  // run on that thread. Destroying a buffer-holding transfer on any other thread would send
  // recycle_now to the wrong shard and corrupt its accounting.
  if (!is_device || buffer.get() == nullptr) { return; }
  try {
    PushAndPopContext c(device_ctx);
    BounceBufferCache::instance().recycle_now(device_ctx, std::move(buffer));
  } catch (...) {
    // A destructor must not throw. If the context push fails (rare), the buffer's own destructor
    // still returns it to the global BounceBufferPool, so there is no memory leak; the only cost is
    // that the shard's checked_out count is not decremented, permanently losing one slot of that
    // shard's capacity.
  }
}

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
      // (1) Splice newly submitted transfers out of the inbox (shared by the reactor thread and
      // submission thread) to minimize the lock duration.
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

      // Admission walk over the reactor-private, lock-free _pending. Entries pop off the front and
      // are either admitted to libcurl or moved to `deferred_transfers`. `_pending` is rebuilt at
      // the end from `deferred_transfers`.
      std::deque<std::unique_ptr<RemoteMultiTransfer>> deferred_transfers;
      // Contexts whose bounce-buffer shard has already missed during this walk. It is assumed that
      // distinct contexts are few, so a flat vector with linear find suffices.
      std::vector<CUcontext> exhausted_ctxs;
      while (!_pending.empty()) {
        auto transfer = std::move(_pending.front());
        _pending.pop_front();
        try {
          // A ctx that already missed the cache this walk cannot admit further device transfers
          // now. Defer without acquiring a limiter slot. The minor cost is that a recycle callback
          // may free a buffer mid-walk, making this pessimistic by one loop iteration at most.
          if (transfer->is_device &&
              std::find(exhausted_ctxs.begin(), exhausted_ctxs.end(), transfer->device_ctx) !=
                exhausted_ctxs.end()) {
            deferred_transfers.push_back(std::move(transfer));
            continue;
          }

          // Gate 1: This gates the network concurrency. Limit the number of HTTP range requests
          // simultaneously attached to this reactor's multi handle for host and device combined.
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
            // Gate 2: This gates the bounce buffer availability. Limit the number of bounce buffers
            // held by this (reactor thread, CUDA context) pair across all pipeline phases. A slot
            // released at libcurl completion does not free a bounce buffer. The buffer stays
            // in-flight until the H2D drains and the cuLaunchHostFunc recycle callback fires.
            std::optional<CudaPinnedBounceBufferPool::Buffer> bounce_buffer;
            {
              PushAndPopContext c(transfer->device_ctx);
              bounce_buffer = BounceBufferCache::instance().try_get(transfer->device_ctx);
            }
            if (!bounce_buffer) {
              exhausted_ctxs.push_back(transfer->device_ctx);
              deferred_transfers.push_back(std::move(transfer));
              continue;
            }
            transfer->buffer            = std::move(*bounce_buffer);
            transfer->ctx.pinned_buffer = transfer->buffer.get();
          }

          CURL* easy    = transfer->curl->handle();
          auto const mc = curl_multi_add_handle(_curl_multi, easy);
          if (mc != CURLM_OK) {
            // The handle was not attached (add failed), so the attachment guard stays inert.
            // Notify the aggregate to satisfy its sub-range count invariant. Null the local pointer
            // so the catch below skips requeueing this already-failed transfer. The buffer
            // auto-recycles via ~RemoteMultiTransfer() when reset() fires.
            transfer->aggregate->on_subrange_failed(std::make_exception_ptr(std::runtime_error(
              std::string("curl_multi_add_handle: ") + curl_multi_strerror(mc))));
            transfer.reset();
            KVIKIO_FAIL(std::string("curl_multi_add_handle: ") + curl_multi_strerror(mc),
                        std::runtime_error);
          }
          // The handle is attached now. Arm the RAII guard so it is removed from the multi handle
          // when the transfer is destroyed, before its CurlHandle returns the easy handle to the
          // pool. Arm before the _in_flight.emplace so an emplace failure still detaches on unwind.
          transfer->attachment = CurlMultiAttachment{_curl_multi, easy};
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
      // The walk drained `_pending`. The deferred entries become the new pending queue.
      std::swap(_pending, deferred_transfers);

      // (2) Drive transfers in a non-blocking way.
      int running_handles   = 0;
      auto const perform_mc = curl_multi_perform(_curl_multi, &running_handles);
      KVIKIO_EXPECT(perform_mc == CURLM_OK,
                    std::string("curl_multi_perform: ") + curl_multi_strerror(perform_mc),
                    std::runtime_error);

      // (3) Drain completions.
      int msgs_left = 0;
      // Set when at least one transfer completes this iteration. Each completed transfer releases
      // its limiter slot when it drops at end of scope, so a completion may unblock a deferred
      // transfer that was waiting on a slot.
      bool completed_any = false;
      while (auto* msg = curl_multi_info_read(_curl_multi, &msgs_left)) {
        if (msg->msg != CURLMSG_DONE) { continue; }
        completed_any = true;
        auto* easy    = msg->easy_handle;
        auto res      = msg->data.result;

        auto it = _in_flight.find(easy);
        KVIKIO_EXPECT(it != _in_flight.end(),
                      "MultiPollReactor: completion for unknown handle",
                      std::runtime_error);
        auto transfer = std::move(it->second);
        _in_flight.erase(it);
        // The easy handle is detached when `transfer` drops at end of scope: transfer->attachment
        // destructs before transfer->curl, so the remove happens before the CurlHandle returns the
        // handle to the LibCurl pool. See CurlMultiAttachment.

        std::exception_ptr transfer_err;
        if (res == CURLE_OK && !transfer->ctx.overflow_error) {
          try {
            if (transfer->is_device) {
              // Phase A (network -> pinned) done. Now schedule Phase B (pinned -> device) on this
              // (thread, ctx) stream and hand the buffer to a cuLaunchHostFunc recycle callback so
              // the cache slot is returned when the H2D drains.
              PushAndPopContext c(transfer->device_ctx);
              CUstream stream = StreamCachePerThreadAndContext::get();
              KVIKIO_CUDA_DRIVER_TRY(
                cudaAPI::instance().MemcpyHtoDAsync(convert_void2deviceptr(transfer->device_dst),
                                                    transfer->buffer.get(),
                                                    transfer->ctx.size,
                                                    stream));
              transfer->aggregate->io_event_barrier->record_event(stream);
              BounceBufferCache::instance().recycle_after(transfer->device_ctx,
                                                          std::move(transfer->buffer),
                                                          stream,
                                                          [curl_multi = _curl_multi]() noexcept {
                                                            std::ignore =
                                                              curl_multi_wakeup(curl_multi);
                                                          });
            }
            transfer->aggregate->on_subrange_complete(transfer->ctx.size);
          } catch (...) {
            transfer_err = std::current_exception();
          }
        } else {
          // Prefer the handle's recorded error buffer. Fall back to the generic strerror text when
          // libcurl recorded no message.
          auto const errmsg = transfer->curl->error_message();
          std::string desc  = std::string("curl_multi transfer failed (") +
                             (errmsg.empty() ? std::string{curl_easy_strerror(res)} : errmsg) + ")";
          if (transfer->ctx.overflow_error) {
            desc += " [server returned more bytes than requested; maybe range support missing?]";
          }
          transfer_err = std::make_exception_ptr(std::runtime_error(std::move(desc)));
        }
        if (transfer_err) { transfer->aggregate->on_subrange_failed(transfer_err); }
        // Transfer (unique_ptr) drops here, returning easy to the LibCurl pool and releasing the
        // transfer's concurrency slot so a deferred request can be admitted. If the buffer was not
        // moved into recycle_after, ~RemoteMultiTransfer() recycles it now.
      }

      // (4) Wait for activity, wakeup, or a bounded timeout. Shorten the timeout while transfers
      // remain deferred in `_pending`, so admission is retried promptly. Recycle callbacks that
      // free a cache slot and completions that free a limiter slot do not necessarily raise libcurl
      // socket activity, so without this we could sleep up to 1s before retrying a deferred
      // transfer.
      // When completions this iteration freed limiter slots and deferred transfers remain, poll
      // without blocking so the next admission walk retries immediately instead of waiting out the
      // 10ms floor. The next iteration reverts to the bounded timeout if it makes no progress, so
      // this does not busy-spin.
      int poll_timeout_ms = _pending.empty() ? 1000 : 10;
      if (completed_any && !_pending.empty()) { poll_timeout_ms = 0; }
      auto const poll_mc = curl_multi_poll(_curl_multi,
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

  // Drain the deferred queue. Deferred transfers normally hold neither a checked-out buffer nor an
  // armed attachment (they were deferred before or during buffer checkout, never after admission).
  // The one exception is a transfer requeued after an _in_flight.emplace failure, which may carry
  // both; destroying it here detaches the handle and recycles the buffer.
  while (!_pending.empty()) {
    auto transfer = std::move(_pending.front());
    _pending.pop_front();
    transfer->aggregate->on_subrange_failed(eptr);
  }

  // In-flight is touched only by the I/O thread, which is us, so no lock needed.
  for (auto& in_flight_entry : _in_flight) {
    in_flight_entry.second->aggregate->on_subrange_failed(eptr);
  }
  // Destroying each transfer detaches its easy handle (transfer->attachment) before its CurlHandle
  // returns the handle to the pool, and recycles its bounce buffer if one was checked out. The
  // residual UB when curl_multi_remove_handle itself fails is documented in ~CurlMultiAttachment.
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
  if (_dispatch == RemoteReactorDispatch::PER_PREAD) {
    auto const idx = _next_reactor_counter.fetch_add(1, std::memory_order_relaxed) % reactor_count;
    _reactors[idx]->submit(std::move(transfers));
    return;
  }

  // PER_CHUNK: round-robin sub-ranges across reactors.
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
