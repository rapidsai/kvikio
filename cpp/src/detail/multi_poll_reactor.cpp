/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <pthread.h>

#include <chrono>
#include <cstddef>
#include <exception>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

#include <curl/curl.h>

#include <kvikio/defaults.hpp>
#include <kvikio/detail/multi_poll_reactor.hpp>
#include <kvikio/error.hpp>
#include <kvikio/remote_handle.hpp>
#include <kvikio/shim/libcurl.hpp>

namespace kvikio::detail {

// ---------------------------------------------------------------------------
// RemoteMultiAggregateContext
// ---------------------------------------------------------------------------

RemoteMultiAggregateContext::RemoteMultiAggregateContext(std::size_t num_subranges)
  : _remaining{num_subranges}
{
  KVIKIO_EXPECT(num_subranges > 0,
                "RemoteMultiAggregateContext requires at least one sub-range",
                std::invalid_argument);
}

void RemoteMultiAggregateContext::on_subrange_complete(std::size_t bytes)
{
  _bytes_acc.fetch_add(bytes, std::memory_order_relaxed);
  // Last thread to decrement to zero fulfills the promise.
  if (_remaining.fetch_sub(1, std::memory_order_acq_rel) == 1) {
    std::lock_guard<std::mutex> lk(_failure_mutex);
    if (_first_exception) {
      _promise.set_exception(_first_exception);
    } else {
      _promise.set_value(_bytes_acc.load(std::memory_order_relaxed));
    }
  }
}

void RemoteMultiAggregateContext::on_subrange_failed(std::exception_ptr ep)
{
  {
    std::lock_guard<std::mutex> lk(_failure_mutex);
    if (!_first_exception) { _first_exception = ep; }
  }
  if (_remaining.fetch_sub(1, std::memory_order_acq_rel) == 1) {
    std::lock_guard<std::mutex> lk(_failure_mutex);
    _promise.set_exception(_first_exception);
  }
}

std::future<std::size_t> RemoteMultiAggregateContext::get_future() { return _promise.get_future(); }

// ---------------------------------------------------------------------------
// MultiPollReactor
// ---------------------------------------------------------------------------

MultiPollReactor::MultiPollReactor()
{
  // Force LibCurl global init before we create the multi handle. (LibCurl is itself a
  // Meyers singleton; we don't care about its destruction order because the pool is
  // leaked and so are we.)
  (void)LibCurl::instance();
  _multi = curl_multi_init();
  KVIKIO_EXPECT(_multi != nullptr, "curl_multi_init() failed", std::runtime_error);
  _io_thread = std::thread(&MultiPollReactor::io_thread_main, this);
}

MultiPollReactor::~MultiPollReactor() noexcept
{
  // Intentionally empty body. `MultiReactorPool` is a leaked-pointer singleton, so its
  // `_reactors` vector and the `std::unique_ptr<MultiPollReactor>` elements inside it
  // are never destroyed. We declare this dtor so the type is complete and usable in
  // `std::unique_ptr`; running it would attempt `~std::thread()` on an unjoined thread
  // and call `std::terminate()`.
}

void MultiPollReactor::submit(std::unique_ptr<RemoteMultiTransfer> transfer)
{
  {
    std::lock_guard<std::mutex> lk(_submit_mutex);
    _pending_add.push_back(std::move(transfer));
  }
  // Thread-safe: the only multi-handle call we make from outside the I/O thread.
  curl_multi_wakeup(_multi);
}

void MultiPollReactor::io_thread_main()
{
  // Best-effort thread naming for profilers and `htop`. Ignored on systems where the
  // name is unavailable. We don't use the reactor index here (the pool sets a more
  // specific name immediately after construction if it wants to); fall back to a
  // generic label.
  pthread_setname_np(pthread_self(), "kvikio-mpoll");

  while (!_stop.load(std::memory_order_acquire)) {
    // (1) Drain submit queue: attach each pending easy handle to the multi.
    {
      std::unique_lock<std::mutex> lk(_submit_mutex);
      while (!_pending_add.empty()) {
        auto t = std::move(_pending_add.front());
        _pending_add.pop_front();
        lk.unlock();
        CURL* easy = t->curl->handle();
        // The caller already set WRITEFUNCTION/WRITEDATA and the endpoint options. We
        // just attach.
        CURLMcode const mc = curl_multi_add_handle(_multi, easy);
        if (mc != CURLM_OK) {
          auto ep = std::make_exception_ptr(
            std::runtime_error(std::string("curl_multi_add_handle: ") + curl_multi_strerror(mc)));
          t->agg->on_subrange_failed(ep);
          // unique_ptr drops here, returning the easy handle to the LibCurl pool.
        } else {
          _in_flight.emplace(easy, std::move(t));
        }
        lk.lock();
      }
    }

    // (2) Drive transfers (non-blocking).
    int running_handles = 0;
    curl_multi_perform(_multi, &running_handles);

    // (3) Drain completions.
    int msgs_left = 0;
    while (CURLMsg* m = curl_multi_info_read(_multi, &msgs_left)) {
      if (m->msg != CURLMSG_DONE) { continue; }
      CURL* easy   = m->easy_handle;
      CURLcode res = m->data.result;

      auto it = _in_flight.find(easy);
      KVIKIO_EXPECT(it != _in_flight.end(),
                    "MultiPollReactor: completion for unknown handle",
                    std::runtime_error);
      auto transfer = std::move(it->second);
      _in_flight.erase(it);
      // *** Critical ordering: remove from multi BEFORE the transfer (and its CurlHandle)
      // is destroyed at end of scope. Otherwise libcurl undefined behavior. ***
      curl_multi_remove_handle(_multi, easy);

      if (res == CURLE_OK && !transfer->ctx.overflow_error) {
        transfer->agg->on_subrange_complete(transfer->ctx.size);
      } else {
        std::stringstream ss;
        ss << "curl_multi transfer failed (" << curl_easy_strerror(res) << ")";
        if (transfer->ctx.overflow_error) {
          ss << " [server returned more bytes than requested; maybe range support "
                "missing?]";
        }
        transfer->agg->on_subrange_failed(std::make_exception_ptr(std::runtime_error(ss.str())));
      }
      // transfer (unique_ptr) drops here, returning easy to the LibCurl pool.
    }

    // (4) Wait for activity, wakeup, or a bounded timeout. The bounded timeout is only
    // useful for test/teardown paths; in production the loop runs until process exit.
    int numfds = 0;
    curl_multi_poll(_multi, nullptr, 0, /*timeout_ms=*/1000, &numfds);
  }

  // Unreachable in production: `_stop` is never set, the loop runs until the OS kills
  // the thread at process exit. Retained for symmetry with future test hooks.
}

// ---------------------------------------------------------------------------
// MultiReactorPool
// ---------------------------------------------------------------------------

MultiReactorPool::MultiReactorPool() : _dispatch{defaults::remote_io_reactor_dispatch()}
{
  // Force LibCurl global init before any reactor opens a multi handle.
  (void)LibCurl::instance();

  auto const n = defaults::remote_io_num_reactors();
  KVIKIO_EXPECT(n > 0, "remote_io_num_reactors must be a positive integer", std::invalid_argument);
  _reactors.reserve(n);
  for (unsigned int i = 0; i < n; ++i) {
    _reactors.emplace_back(std::make_unique<MultiPollReactor>());
  }
}

MultiReactorPool::~MultiReactorPool() noexcept
{
  // Intentionally empty body. The pool is a leaked-pointer singleton (see `instance()`),
  // so this destructor is never invoked.
}

MultiReactorPool& MultiReactorPool::instance()
{
  // Heap-leaked singleton: matches the convention used by `BounceBufferPool` and
  // `StreamCachePerThreadAndContext`. The pool, its reactors, and their `std::thread`s
  // are never destroyed; the OS reclaims them at process exit.
  static MultiReactorPool* inst = new MultiReactorPool();
  return *inst;
}

void MultiReactorPool::submit_pread(std::vector<std::unique_ptr<RemoteMultiTransfer>> transfers)
{
  auto const n = _reactors.size();
  KVIKIO_EXPECT(n > 0, "MultiReactorPool has no reactors", std::runtime_error);

  if (_dispatch == RemoteReactorDispatch::PER_PREAD) {
    // One reactor for the whole pread() call: preserves per-CURLM connection-pool reuse.
    auto const idx = _pread_counter.fetch_add(1, std::memory_order_relaxed) % n;
    for (auto& t : transfers) {
      _reactors[idx]->submit(std::move(t));
    }
    return;
  }

  // PER_CHUNK (default): round-robin sub-ranges across reactors.
  for (auto& t : transfers) {
    auto const idx = _chunk_counter.fetch_add(1, std::memory_order_relaxed) % n;
    _reactors[idx]->submit(std::move(t));
  }
}

}  // namespace kvikio::detail
