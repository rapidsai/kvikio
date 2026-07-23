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
#include <kvikio/logger.hpp>
#include <kvikio/logger_macros.hpp>
#include <kvikio/remote_handle.hpp>
#include <kvikio/shim/cuda.hpp>
#include <kvikio/shim/libcurl.hpp>
#include <kvikio/utils.hpp>

namespace kvikio::detail {

namespace {
void detach_from_multi(CURLM* multi, CURL* easy) noexcept
{
  if (multi == nullptr || easy == nullptr) { return; }
  auto const mc = curl_multi_remove_handle(multi, easy);
  if (mc != CURLM_OK) {
    KVIKIO_LOG_ERROR(std::string("CurlMultiAttachment: curl_multi_remove_handle failed: ") +
                     curl_multi_strerror(mc));
  }
}
}  // namespace

CurlMultiAttachment::CurlMultiAttachment(CURLM* multi, CURL* easy) noexcept
  : _multi{multi}, _easy{easy}
{
}

CurlMultiAttachment::~CurlMultiAttachment()
{
  // Best-effort detach on the reactor I/O thread. If curl_multi_remove_handle fails (rare), the
  // handle stays attached and the owning CurlHandle still returns it to the LibCurl pool, which is
  // undefined behavior in libcurl. A destructor has no better recovery.
  detach_from_multi(_multi, _easy);
}

CurlMultiAttachment::CurlMultiAttachment(CurlMultiAttachment&& o) noexcept
  : _multi{std::exchange(o._multi, nullptr)}, _easy{std::exchange(o._easy, nullptr)}
{
}

CurlMultiAttachment& CurlMultiAttachment::operator=(CurlMultiAttachment&& o) noexcept
{
  if (this != &o) {
    // Detach whatever this guard currently holds before taking over o's handle.
    detach_from_multi(_multi, _easy);
    _multi = std::exchange(o._multi, nullptr);
    _easy  = std::exchange(o._easy, nullptr);
  }
  return *this;
}

RemoteMultiTransfer::~RemoteMultiTransfer()
{
  using BounceBufferCache = BounceBufferCachePerThreadAndContext<CudaPinnedAllocator>;
  // A device transfer still holding its bounce buffer reaches here only on a failure path. The
  // success path moves the buffer into recycle_after, leaving buffer.get() == nullptr.
  //
  // Thread-affinity invariant: the cache is sharded by (this_thread::get_id(), ctx), so a
  // buffer-holding transfer MUST be destroyed on the reactor I/O thread that checked it out in
  // stage (1). Every such destruction (in-flight drain, admission-walk reset(), completion drop)
  // runs on that thread. Destroying it elsewhere would recycle into the wrong shard and corrupt
  // that shard's accounting.
  if (!is_device || buffer.get() == nullptr) { return; }
  try {
    PushAndPopContext c(device_ctx);
    BounceBufferCache::instance().recycle_now(device_ctx, std::move(buffer));
  } catch (std::exception const& e) {
    KVIKIO_LOG_ERROR(std::string("RemoteMultiTransfer: buffer recycle failed: ") + e.what());
  } catch (...) {
    KVIKIO_LOG_ERROR("RemoteMultiTransfer: buffer recycle failed: unknown exception");
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
  // The last thread to decrement _subranges_left to zero fulfills the promise. Its acq_rel
  // decrement acquires every other thread's relaxed _total_bytes writes (each released by that
  // thread's own decrement), so the sum is complete. _first_exception needs no ordering here, since
  // it is written and read under _exception_mutex.
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
  // Intentionally empty. Reactors are owned by the leaked `MultiReactorPool` singleton and never
  // destroyed. This dtor exists only to complete the type for `std::unique_ptr`. Running it would
  // destroy an unjoined `std::thread` and call `std::terminate()`.
}

void MultiPollReactor::wakeup() noexcept { std::ignore = curl_multi_wakeup(_curl_multi); }

void MultiPollReactor::submit(std::vector<std::unique_ptr<RemoteMultiTransfer>> transfers)
{
  if (transfers.empty()) { return; }
  std::exception_ptr fail_reason;
  {
    std::lock_guard<std::mutex> const lock(_submit_mutex);
    if (_pool->is_dead()) {
      // The pool is dead. Fail the batch immediately instead of pushing into an inbox that will
      // never be drained.
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
      // Stage (1): Splice newly submitted transfers out of the inbox (shared by the reactor thread
      // and submission thread) to minimize the lock duration.
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

      // Admission walk over the reactor-private _pending. Each entry is either admitted to libcurl
      // or moved to `deferred_transfers`, which becomes the new `_pending` at the end.
      std::deque<std::unique_ptr<RemoteMultiTransfer>> deferred_transfers;
      // Contexts whose bounce-buffer shard has already missed during this walk. It is assumed that
      // distinct contexts are few, so a flat vector with linear find suffices.
      std::vector<CUcontext> exhausted_ctxs;
      while (!_pending.empty()) {
        auto transfer = std::move(_pending.front());
        _pending.pop_front();
        try {
          // This ctx already missed the cache this walk, so defer without taking a limiter slot. At
          // worst this is pessimistic by one iteration if a recycle frees a buffer mid-walk.
          if (transfer->is_device &&
              std::find(exhausted_ctxs.begin(), exhausted_ctxs.end(), transfer->device_ctx) !=
                exhausted_ctxs.end()) {
            deferred_transfers.push_back(std::move(transfer));
            continue;
          }

          // Gate 1 caps network concurrency. Limit the HTTP range requests attached to this
          // reactor's multi handle at once, host and device combined.
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
            // Gate 2 caps bounce-buffer use per (reactor thread, CUDA context) across all pipeline
            // phases. A limiter slot freed at libcurl completion does not free the buffer, which
            // stays in-flight until the H2D drains and the recycle callback fires.
            std::optional<CudaPinnedBounceBufferPool::Buffer> bounce_buffer;
            {
              PushAndPopContext c(transfer->device_ctx);
              bounce_buffer = BounceBufferCache::instance().try_get(transfer->device_ctx);
            }
            if (!bounce_buffer.has_value()) {
              exhausted_ctxs.push_back(transfer->device_ctx);
              deferred_transfers.push_back(std::move(transfer));
              continue;
            }
            transfer->buffer            = std::move(bounce_buffer.value());
            transfer->ctx.pinned_buffer = transfer->buffer.get();
          }

          CURL* easy    = transfer->curl->handle();
          auto const mc = curl_multi_add_handle(_curl_multi, easy);
          if (mc != CURLM_OK) {
            transfer->aggregate->on_subrange_failed(std::make_exception_ptr(std::runtime_error(
              std::string("curl_multi_add_handle: ") + curl_multi_strerror(mc))));
            transfer.reset();
            KVIKIO_FAIL(std::string("curl_multi_add_handle: ") + curl_multi_strerror(mc),
                        std::runtime_error);
          }
          transfer->attachment = CurlMultiAttachment{_curl_multi, easy};
          transfer->slot       = std::move(slot);
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

      // Stage (2): Drive transfers in a non-blocking way.
      int running_handles   = 0;
      auto const perform_mc = curl_multi_perform(_curl_multi, &running_handles);
      KVIKIO_EXPECT(perform_mc == CURLM_OK,
                    std::string("curl_multi_perform: ") + curl_multi_strerror(perform_mc),
                    std::runtime_error);

      // Stage (3): Drain completions.
      int msgs_left = 0;
      // A completion frees a limiter slot, which may unblock a deferred transfer waiting on one.
      // Stage (4) uses this to shorten the poll timeout.
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
      }

      // Stage (4): Wait for socket activity, a wakeup, or a timeout. Limiter-slot and bounce-buffer
      // frees do not raise socket activity, so the timeout adapts. It is 1s when idle, 10ms while
      // transfers stay deferred in `_pending`, and 0 when a completion this iteration freed a slot
      // and work remains, so admission retries at once.
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
    KVIKIO_LOG_ERROR("MultiPollReactor: fatal libcurl error, reactor pool declared dead");
    _pool->signal_death(std::current_exception());
  }
  // Reached by catching the exception above or by noticing _pool->is_dead() at the loop top. Either
  // way, drain our own state with the recorded reason so no caller's future.get() hangs.
  fail_all_pending(_pool->death_reason());
}

void MultiPollReactor::fail_all_pending(std::exception_ptr eptr)
{
  // Drain the inbox under the submit mutex.
  {
    std::lock_guard<std::mutex> const lock(_submit_mutex);
    while (!_inbox.empty()) {
      auto transfer = std::move(_inbox.front());
      _inbox.pop_front();
      transfer->aggregate->on_subrange_failed(eptr);
    }
  }

  // Drain the deferred queue.
  while (!_pending.empty()) {
    auto transfer = std::move(_pending.front());
    _pending.pop_front();
    transfer->aggregate->on_subrange_failed(eptr);
  }

  // In-flight is touched only by the I/O thread, which is us, so no lock needed.
  for (auto& in_flight_entry : _in_flight) {
    in_flight_entry.second->aggregate->on_subrange_failed(eptr);
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
  // Intentionally empty. The pool is a leaked singleton, so this dtor is never invoked.
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
  // The lock serializes _death_reason writes and keeps the _dead store in its scope so the first
  // writer wins, not the last. The store is `release`, pairing with the `acquire` in `is_dead()`.
  // The guard load below can be relaxed.
  {
    std::lock_guard<std::mutex> const lock(_death_mutex);
    // Only the first thread here updates _death_reason and wakes reactors. Later calls early-exit.
    if (_dead.load(std::memory_order_relaxed)) { return; }
    _death_reason = eptr;
    _dead.store(true, std::memory_order_release);
  }

  // Wake every reactor out of curl_multi_poll so they notice _dead promptly. Including the caller's
  // own reactor is harmless, since it has already left its loop.
  for (auto const& r : _reactors) {
    r->wakeup();
  }
}

}  // namespace kvikio::detail
