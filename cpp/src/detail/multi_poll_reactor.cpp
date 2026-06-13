/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cstddef>
#include <exception>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>

#include <curl/curl.h>

#include <kvikio/defaults.hpp>
#include <kvikio/detail/multi_poll_reactor.hpp>
#include <kvikio/error.hpp>
#include <kvikio/remote_handle.hpp>
#include <kvikio/shim/libcurl.hpp>

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

MultiPollReactor::MultiPollReactor(MultiReactorPool* pool, std::size_t max_concurrent_requests)
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
  try {
    while (!_pool->is_dead()) {
      // Set true if stage (1) leaves at least one request queued because the concurrency limiter
      // is at capacity. Stage (4) then shortens the poll timeout so the inbox is re-checked
      // promptly once a completing request frees a slot (a release on another reactor thread does
      // not raise libcurl socket activity here).
      bool has_deferred = false;

      // (1) Drain submission queue. Admit as many requests as the concurrency limiter allows and
      // attach each admitted easy handle to the multi handle.
      {
        // Lock is needed since _inbox is shared between the I/O thread and the caller thread.
        std::unique_lock<std::mutex> lock(_submit_mutex);
        while (!_inbox.empty()) {
          // Bound this reactor's in-flight requests to its private share of the global budget. Once
          // its limiter is at capacity every remaining inbox entry would fail the same check, so
          // stop the walk and leave the rest queued in FIFO order for a later iteration. Because the
          // share is private, completions on this reactor free its own slots and it re-admits its
          // own inbox with no cross-reactor hand-off.
          if (!_request_limiter.try_acquire()) {
            has_deferred = true;
            break;
          }
          auto transfer = std::move(_inbox.front());
          _inbox.pop_front();
          lock.unlock();

          CURL* easy = transfer->curl->handle();
          // The caller already set WRITEFUNCTION/WRITEDATA and the endpoint options for the easy
          // handle. We just attach it to the multi handle.
          auto const mc = curl_multi_add_handle(_curl_multi, easy);
          if (mc != CURLM_OK) {
            // The slot was acquired for a transfer that will never reach _in_flight, so release it
            // now to avoid permanently shrinking capacity.
            _request_limiter.release();
            // This transfer is now in transit between _inbox and _in_flight. fail_all_pending (run
            // on the catch path below) iterates only those two containers, so it cannot find this
            // transfer. We must mark its aggregate as failed here to maintain the per-aggregate
            // sub-range count invariant. Otherwise the aggregate's _promise would not be resolved,
            // and the caller's future.get() would hang.
            transfer->aggregate->on_subrange_failed(std::make_exception_ptr(std::runtime_error(
              std::string("curl_multi_add_handle: ") + curl_multi_strerror(mc))));
            KVIKIO_FAIL(std::string("curl_multi_add_handle: ") + curl_multi_strerror(mc),
                        std::runtime_error);
          }
          _in_flight.emplace(easy, std::move(transfer));
          lock.lock();
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
          transfer->aggregate->on_subrange_complete(transfer->ctx.size);
        } else {
          std::stringstream ss;
          ss << "curl_multi transfer failed (" << curl_easy_strerror(res) << ")";
          if (transfer->ctx.overflow_error) {
            ss << " [server returned more bytes than requested; maybe range support "
                  "missing?]";
          }
          transfer->aggregate->on_subrange_failed(
            std::make_exception_ptr(std::runtime_error(ss.str())));
        }
        // This request has left _in_flight (success or failure); return its concurrency slot so a
        // deferred request can be admitted.
        _request_limiter.release();
        // transfer (unique_ptr) drops here, returning easy to the LibCurl pool.
      }

      // (4) Wait for activity, wakeup, or a bounded timeout. Shorten the timeout when at least one
      // request is deferred on a full limiter, so the inbox is re-checked promptly after a
      // completing request frees a slot (such a release does not raise libcurl socket activity).
      int const poll_timeout_ms = has_deferred ? 10 : 1000;
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
  // Drain the inbox under the submit mutex. New submissions are blocked from accumulating by the
  // _pool->is_dead() check in submit(), which is already true by the time we get here.
  {
    std::lock_guard<std::mutex> const lock(_submit_mutex);
    while (!_inbox.empty()) {
      auto transfer = std::move(_inbox.front());
      _inbox.pop_front();
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
    transfer->aggregate->on_subrange_failed(eptr);
    // Every _in_flight entry holds a concurrency slot (the inbox entries drained above never
    // acquired one, so they are not released here).
    _request_limiter.release();
  }
  _in_flight.clear();
}

MultiReactorPool::MultiReactorPool() : _dispatch{defaults::remote_io_reactor_dispatch()}
{
  // Force LibCurl global init before any reactor opens a multi handle.
  std::ignore = LibCurl::instance();

  auto const n = defaults::remote_io_num_reactors();
  KVIKIO_EXPECT(n > 0, "remote_io_num_reactors must be a positive integer", std::invalid_argument);

  // Split the global concurrent-request budget into a private per-reactor share. Each reactor
  // enforces its own share against its own inbox, which avoids a single shared admission gate (that
  // gate caused reactor starvation and pipeline stalls under heavy re-admission churn). The shares
  // sum to at most the global cap, so total in-flight stays bounded. 0 means unlimited. Floor the
  // share at 1 so a cap smaller than the reactor count cannot starve a reactor into a hang (this
  // can let the sum slightly exceed the cap only in that small-cap corner case).
  auto const max_total = defaults::remote_io_max_concurrent_requests();
  std::size_t per_reactor_max = (max_total == 0) ? 0 : (max_total / n);
  if (max_total != 0 && per_reactor_max == 0) { per_reactor_max = 1; }

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
