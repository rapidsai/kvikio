/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <exception>
#include <limits>
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
#include <kvikio/statistics/counters.hpp>
#include <kvikio/utils.hpp>

namespace kvikio::detail {

CurlMultiAttachment::CurlMultiAttachment(CURLM* multi, CURL* easy) noexcept
  : _multi{multi}, _easy{easy}
{
}

void CurlMultiAttachment::reset() noexcept
{
  if (_multi != nullptr && _easy != nullptr) {
    // Best-effort detach on the reactor I/O thread. If curl_multi_remove_handle fails (rare), the
    // handle stays attached and the owning CurlHandle still returns it to the LibCurl pool, which
    // is undefined behavior in libcurl. There is no better recovery available here.
    auto const mc = curl_multi_remove_handle(_multi, _easy);
    if (mc != CURLM_OK) {
      KVIKIO_LOG_ERROR(std::string("CurlMultiAttachment: curl_multi_remove_handle failed: ") +
                       curl_multi_strerror(mc));
    }
  }
  _multi = nullptr;
  _easy  = nullptr;
}

CurlMultiAttachment::~CurlMultiAttachment() { reset(); }

CurlMultiAttachment::CurlMultiAttachment(CurlMultiAttachment&& other) noexcept
  : _multi{std::exchange(other._multi, nullptr)}, _easy{std::exchange(other._easy, nullptr)}
{
}

CurlMultiAttachment& CurlMultiAttachment::operator=(CurlMultiAttachment&& other) noexcept
{
  if (this != &other) {
    // Detach whatever this guard currently holds before taking over o's handle.
    reset();
    _multi = std::exchange(other._multi, nullptr);
    _easy  = std::exchange(other._easy, nullptr);
  }
  return *this;
}

RemoteMultiTransfer::~RemoteMultiTransfer()
{
  using BounceBufferCache = BounceBufferCachePerThreadAndContext<CudaPinnedAllocator>;
  // A device transfer still holding its bounce buffer reaches here only on a failure path. The
  // success path moves the buffer into recycle_after, leaving buffer.get() == nullptr.
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
    // Finish the observation before fulfilling the promise below. The other order would let the
    // caller return from `future.get()` before the observation had been delivered.
    if (recorder) {
      if (_first_exception) {
        recorder->finish_with_failure();
      } else {
        recorder->finish(_total_bytes.load(std::memory_order_relaxed));
      }
    }
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
    if (recorder) { recorder->finish_with_failure(); }
    _promise.set_exception(_first_exception);
  }
}

std::future<std::size_t> RemoteMultiAggregateContext::get_future() { return _promise.get_future(); }

MultiPollReactor::MultiPollReactor(MultiReactorPool* pool,
                                   std::optional<std::size_t> max_concurrent_requests,
                                   ConcurrentRequestLimiter* shared_limiter)
  : _pool{pool},
    _private_limiter{max_concurrent_requests},
    _request_limiter{shared_limiter != nullptr ? shared_limiter : &_private_limiter},
    _take_ceiling{max_concurrent_requests.has_value()
                    ? std::max<std::size_t>(max_concurrent_requests.value() * 2, 2)
                    : std::numeric_limits<std::size_t>::max()}
{
  KVIKIO_EXPECT(
    _pool != nullptr, "MultiPollReactor requires a non-null pool", std::invalid_argument);
  // Force LibCurl global init before we create the multi handle.
  std::ignore = LibCurl::instance();
  _curl_multi = curl_multi_init();
  KVIKIO_EXPECT(_curl_multi != nullptr, "curl_multi_init() failed", std::runtime_error);
  set_connection_cache_size(max_concurrent_requests);
  _io_thread = std::thread(&MultiPollReactor::io_thread_main, this);
}

std::optional<long> connection_cache_size(
  std::optional<std::size_t> max_concurrent_requests) noexcept
{
  if (!max_concurrent_requests.has_value()) { return std::nullopt; }

  // libcurl documents this option as taking a `long`, and the value is internally stored as an
  // `unsigned int`. So we cap at whichever of UINT_MAX and LONG_MAX is smaller.
  constexpr auto uint_max = static_cast<std::size_t>(std::numeric_limits<unsigned>::max());
  constexpr auto long_max = static_cast<std::size_t>(std::numeric_limits<long>::max());
  constexpr std::size_t max_settable = std::min(uint_max, long_max);

  // min(max_concurrent_requests * headroom_scale, max_settable), with int overflow avoidance
  constexpr std::size_t headroom_scale = 4;
  auto const max_req_adjusted          = std::max<std::size_t>(max_concurrent_requests.value(), 1);
  auto const tmp = std::min<std::size_t>(max_req_adjusted, max_settable / headroom_scale);
  return static_cast<long>(tmp * headroom_scale);
}

std::optional<std::size_t> bounce_buffer_cap()
{
  auto const max_total = defaults::remote_io_max_concurrent_requests();
  if (max_total == 0) { return std::nullopt; }
  if (defaults::remote_io_reactor_dispatch() == RemoteReactorDispatch::FIRST_AVAILABLE) {
    return max_total;
  }
  auto const n = defaults::remote_io_num_reactors();
  return std::max<std::size_t>(max_total / n, 1);
}

void MultiPollReactor::set_connection_cache_size(
  std::optional<std::size_t> max_concurrent_requests) const
{
  auto const cache_size = connection_cache_size(max_concurrent_requests);
  if (!cache_size.has_value()) { return; }

  auto const mc = curl_multi_setopt(_curl_multi, CURLMOPT_MAXCONNECTS, cache_size.value());
  KVIKIO_EXPECT(mc == CURLM_OK,
                std::string("curl_multi_setopt(CURLMOPT_MAXCONNECTS): ") + curl_multi_strerror(mc),
                std::runtime_error);
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

void MultiPollReactor::AdmitOutcome::merge(AdmitOutcome const& other) noexcept
{
  deferred_for_resource |= other.deferred_for_resource;
  if (!other.earliest_ready_at.has_value()) { return; }
  earliest_ready_at = earliest_ready_at.has_value()
                        ? std::min(earliest_ready_at.value(), other.earliest_ready_at.value())
                        : other.earliest_ready_at;
}

MultiPollReactor::AdmitOutcome MultiPollReactor::admit_pending()
{
  using BounceBufferCache = BounceBufferCachePerThreadAndContext<CudaPinnedAllocator>;

  // Iterate the per-reactor _pending: Each entry is either admitted to libcurl or moved to
  // `deferred_transfers`, which becomes the new `_pending` at the end.
  std::deque<std::unique_ptr<RemoteMultiTransfer>> deferred_transfers;
  // Contexts whose bounce-buffer shard has already missed during this walk. It is assumed that
  // distinct contexts are few, so a flat vector with linear find suffices.
  std::vector<CUcontext> exhausted_ctxs;
  AdmitOutcome outcome;
  auto const walk_start = std::chrono::steady_clock::now();
  while (!_pending.empty()) {
    auto transfer = std::move(_pending.front());
    _pending.pop_front();
    try {
      // Defer a transfer if it is still serving its backoff for retry.
      if (transfer->ready_at > walk_start) {
        if (outcome.earliest_ready_at.has_value()) {
          outcome.earliest_ready_at =
            std::min(outcome.earliest_ready_at.value(), transfer->ready_at);
        } else {
          outcome.earliest_ready_at = transfer->ready_at;
        }
        deferred_transfers.push_back(std::move(transfer));
        continue;
      }

      // This ctx already missed the cache this walk, so defer without taking a limiter slot. At
      // worst this is pessimistic by one iteration if a recycle frees a buffer mid-walk.
      if (transfer->is_device &&
          std::find(exhausted_ctxs.begin(), exhausted_ctxs.end(), transfer->device_ctx) !=
            exhausted_ctxs.end()) {
        outcome.deferred_for_resource = true;
        deferred_transfers.push_back(std::move(transfer));
        continue;
      }

      // Gate 1 caps network concurrency. Limit the HTTP range requests attached to this
      // reactor's multi handle at once, host and device combined. A transfer taken off the
      // pool-wide queue arrives with its reservation already made.
      auto slot = transfer->slot ? std::move(transfer->slot) : _request_limiter->try_acquire();
      if (!slot) {
        outcome.deferred_for_resource = true;
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
          outcome.deferred_for_resource = true;
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
        transfer->aggregate->on_subrange_failed(std::make_exception_ptr(
          std::runtime_error(std::string("curl_multi_add_handle: ") + curl_multi_strerror(mc))));
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
  return outcome;
}

void MultiPollReactor::take_from_pool_queue(AdmitOutcome& outcome)
{
  // Reserve before popping. A sub-range then leaves the queue only when its reactor can start it
  // immediately.
  //
  // Both limits below matter. Without `_take_ceiling` the first reactor to wake takes the entire
  // pool-wide budget and the rest find an empty limiter. Without the fair share one reactor
  // sweeps up the tail of a burst and runs it alone while the others idle.
  auto const take_limit = _pool->fair_take_count();
  std::size_t taken     = 0;
  while (_pool->has_queued_work() && taken < take_limit &&
         _in_flight.size() + _pending.size() < _take_ceiling) {
    auto slot = _request_limiter->try_acquire();
    if (!slot) {
      outcome.deferred_for_resource = true;
      return;
    }
    auto transfer = _pool->try_pop_queued();
    // Another reactor got there first. `slot` returns the reservation as it goes out of scope.
    if (!transfer) { return; }
    transfer->slot = std::move(slot);
    _pending.push_back(std::move(transfer));
    ++taken;
  }
}

void MultiPollReactor::io_thread_main()
{
  using BounceBufferCache = BounceBufferCachePerThreadAndContext<CudaPinnedAllocator>;

  // Whether to re-run the admission walk with the slots that this pass's completions just freed,
  // instead of returning from the poll immediately and noticing them on the next pass. Worth it
  // for FIRST_AVAILABLE, where stage (3b) has just taken queued work that would otherwise sit in
  // `_pending` until the poll times out. For the pre-binding modes it measured inside the
  // run-to-run spread, so they keep the simpler path.
  bool const readmit_after_completion = _pool->uses_first_available();

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

      auto admit = admit_pending();
      if (_pool->uses_first_available()) {
        take_from_pool_queue(admit);
        if (!_pending.empty()) { admit.merge(admit_pending()); }
      }

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
      std::size_t completed_count = 0;
      while (auto* msg = curl_multi_info_read(_curl_multi, &msgs_left)) {
        if (msg->msg != CURLMSG_DONE) { continue; }
        ++completed_count;
        auto* easy = msg->easy_handle;
        auto res   = msg->data.result;

        auto it = _in_flight.find(easy);
        KVIKIO_EXPECT(it != _in_flight.end(),
                      "MultiPollReactor: completion for unknown handle",
                      std::runtime_error);
        auto transfer = std::move(it->second);
        _in_flight.erase(it);
        count_http_connection_of(easy);

        std::exception_ptr transfer_err;
        try {
          if (res == CURLE_OK && !transfer->ctx.overflow_error) {
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
          } else if (transfer->ctx.overflow_error) {
            // Prefer the handle's recorded error buffer. Fall back to the generic strerror text
            // when libcurl recorded no message.
            auto const errmsg = transfer->curl->error_message();
            std::string desc  = std::string("curl_multi transfer failed (") +
                               (errmsg.empty() ? std::string{curl_easy_strerror(res)} : errmsg) +
                               ") [server returned more bytes than requested; maybe range support "
                               "missing?]";
            transfer_err = std::make_exception_ptr(std::runtime_error(std::move(desc)));
          } else {
            long http_code = 0;
            transfer->curl->getinfo(CURLINFO_RESPONSE_CODE, &http_code);
            ++transfer->attempt;
            auto const errmsg  = transfer->curl->error_message();
            auto const outcome = transfer->retry_policy->evaluate(
              res, http_code, transfer->attempt, errmsg, "curl_multi transfer failed");

            if (outcome.decision == RetryDecision::RETRY) {
              KVIKIO_LOG_WARN(outcome.message);
              count_http_retry(outcome.delay_ms);
              auto const ready_at = std::chrono::steady_clock::now() + outcome.delay_ms;
              // If a shorter backoff appears
              if (admit.earliest_ready_at.has_value()) {
                admit.earliest_ready_at = std::min(admit.earliest_ready_at.value(), ready_at);
              } else {
                admit.earliest_ready_at = ready_at;
              }
              requeue_for_retry(std::move(transfer), ready_at);
              continue;
            }

            transfer_err = std::make_exception_ptr(std::runtime_error(outcome.message));
          }
        } catch (...) {
          transfer_err = std::current_exception();
        }
        if (transfer_err) { transfer->aggregate->on_subrange_failed(transfer_err); }
      }

      // Stage (3b): Spend the slots those completions just released now. Waiting for another poll
      // and trip round the loop would leave this reactor holding fewer requests in flight than the
      // gates allow.
      if (completed_count > 0 && _pool->uses_first_available()) { take_from_pool_queue(admit); }

      // A pool-wide slot released here is most likely useful to a different reactor, and nothing
      // else would tell it. Wake as many reactors as there is queued work for rather than as many
      // as completed, because one completion can release capacity that several idle reactors
      // could each use, and each of them takes at least one sub-range.
      if (completed_count > 0 && _pool->uses_first_available()) {
        _pool->wake_reactors(_pool->queued_count_hint());
      }

      if (readmit_after_completion && completed_count > 0 && !_pending.empty()) {
        admit.merge(admit_pending());
        int readmit_handles   = 0;
        auto const readmit_mc = curl_multi_perform(_curl_multi, &readmit_handles);
        KVIKIO_EXPECT(readmit_mc == CURLM_OK,
                      std::string("curl_multi_perform: ") + curl_multi_strerror(readmit_mc),
                      std::runtime_error);
      }

      // Stage (4): Wait for socket activity, a wakeup, a timeout, or elapsed backoff for retry.
      constexpr int idle_timeout_ms = 1000;
      // Backstop for a wakeup that never arrives while work waits on a limiter slot or a bounce
      // buffer. Both are normally released by events that already wake the poll, namely a
      // completion or the recycle callback's `curl_multi_wakeup`.
      constexpr int busy_timeout_ms = 10;
      // Under FIRST_AVAILABLE an empty `_pending` does not mean there is nothing to do. Work waits
      // in the pool-wide queue until some reactor takes it, so idling for a full second here would
      // leave it there until a wakeup happened to pick this reactor.
      bool const pool_work_waiting = _pool->uses_first_available() && _pool->has_queued_work();
      int poll_timeout_ms{};
      if (_pending.empty() && !pool_work_waiting) {
        // Nothing queued here or pool-wide
        poll_timeout_ms = idle_timeout_ms;
      } else if (!admit.deferred_for_resource && admit.earliest_ready_at.has_value()) {
        // Wait for the earliest elapsed backoff, not a limiter slot or bounce buffer resource
        auto const wait_ms = std::chrono::ceil<std::chrono::milliseconds>(
                               admit.earliest_ready_at.value() - std::chrono::steady_clock::now())
                               .count();
        if (wait_ms <= 0) {
          poll_timeout_ms = 0;
        } else if (wait_ms >= idle_timeout_ms) {
          poll_timeout_ms = idle_timeout_ms;
        } else {
          poll_timeout_ms = static_cast<int>(wait_ms);
        }
      } else if (!readmit_after_completion && completed_count > 0) {
        // Only reachable with stage (3b) switched off. A completion's freed slots are then still
        // unspent, and the loop has to come straight back rather than wait.
        poll_timeout_ms = 0;
      } else {
        // Still waiting on a limiter slot or a bounce buffer. Stage (3b) has already retried
        // admission with whatever this pass released. Anything left is genuinely blocked, and
        // returning immediately would spin without making progress.
        poll_timeout_ms = busy_timeout_ms;
      }
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

void MultiPollReactor::requeue_for_retry(std::unique_ptr<RemoteMultiTransfer> transfer,
                                         std::chrono::steady_clock::time_point ready_at) noexcept
{
  using BounceBufferCache = BounceBufferCachePerThreadAndContext<CudaPinnedAllocator>;

  // Extend the lifetime of aggregate (a shared pointer).
  auto aggregate = transfer->aggregate;

  try {
    transfer->attachment.reset();
    transfer->slot.reset();

    if (transfer->is_device && transfer->buffer.get() != nullptr) {
      PushAndPopContext c(transfer->device_ctx);
      BounceBufferCache::instance().recycle_now(transfer->device_ctx, std::move(transfer->buffer));
      transfer->ctx.pinned_buffer = nullptr;
    }

    transfer->ctx.reset_for_retry();
    transfer->curl->clear_error_message();
    transfer->ready_at = ready_at;
    _pending.push_back(std::move(transfer));
  } catch (...) {
    aggregate->on_subrange_failed(std::current_exception());
  }
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

MultiReactorPool::MultiReactorPool()
  : _reactor_count{defaults::remote_io_num_reactors()},
    _dispatch{defaults::remote_io_reactor_dispatch()}
{
  // Force LibCurl global init before any reactor opens a multi handle.
  std::ignore = LibCurl::instance();

  auto const n = _reactor_count;
  KVIKIO_EXPECT(n > 0, "remote_io_num_reactors must be a positive integer", std::invalid_argument);

  auto const max_total = defaults::remote_io_max_concurrent_requests();

  // FIRST_AVAILABLE is paced by the concurrency budget. With no budget every reactor reserves
  // without limit, letting the first to run drain the queue, which is worse than round-robin.
  if (_dispatch == RemoteReactorDispatch::FIRST_AVAILABLE && max_total == 0) {
    KVIKIO_LOG_WARN(
      "KVIKIO_REMOTE_IO_REACTOR_DISPATCH=first_available needs a non-zero "
      "KVIKIO_REMOTE_IO_MAX_CONCURRENT_REQUESTS to pace the queue. Falling back to per_chunk.");
    _dispatch = RemoteReactorDispatch::PER_CHUNK;
  }

  std::optional<std::size_t> const per_reactor_max =
    (max_total == 0) ? std::nullopt : std::optional{std::max<std::size_t>(max_total / n, 1)};

  // One budget for the whole pool rather than a slice each. A slice strands budget in idle
  // reactors and cannot represent a total that does not divide by the reactor count. Rounding
  // down, 512 over 48 reactors yields 10 each, leaving only 480 slots usable.
  //
  // Sound only when nothing binds work to a reactor before a slot exists for it, which is what
  // FIRST_AVAILABLE guarantees. PER_CHUNK and PER_PREAD bind a sub-range at submission, letting a
  // busy reactor take slots another one needs for work it already holds. That reactor stalls and
  // the `pread()` waits on it, which measured slower and far less predictable than slicing.
  // `_dispatch` rather than the configured value, so the fallback above is honored.
  if (_dispatch == RemoteReactorDispatch::FIRST_AVAILABLE) {
    _shared_limiter.emplace(max_total == 0 ? std::nullopt : std::optional{max_total});
  } else if (max_total != 0 && per_reactor_max.value() * n != max_total) {
    KVIKIO_LOG_WARN("KVIKIO_REMOTE_IO_MAX_CONCURRENT_REQUESTS=" + std::to_string(max_total) +
                    " does not divide by KVIKIO_REMOTE_IO_NUM_REACTORS=" + std::to_string(n) +
                    ", so the per-reactor slices allow only " +
                    std::to_string(per_reactor_max.value() * n) +
                    " concurrent requests. Use a multiple of the reactor count, or select "
                    "KVIKIO_REMOTE_IO_REACTOR_DISPATCH=first_available, which enforces the budget "
                    "pool-wide.");
  }

  // These are derived rather than configured, and getting them wrong costs bandwidth without
  // raising an error. Report them once to make a run checkable against what was intended.
  std::string budget_desc{"unlimited"};
  if (max_total != 0) {
    budget_desc = std::to_string(max_total);
    budget_desc += _shared_limiter.has_value()
                     ? " pool-wide"
                     : " sliced into " + std::to_string(per_reactor_max.value()) + " per reactor";
  }
  KVIKIO_LOG_INFO("MULTI_POLL: " + std::to_string(n) + " reactors, concurrency budget " +
                  budget_desc + ", connection cache " +
                  std::to_string(connection_cache_size(per_reactor_max).value_or(0)) +
                  " per reactor");

  auto* const shared = _shared_limiter.has_value() ? &_shared_limiter.value() : nullptr;
  _reactors.reserve(n);
  for (unsigned int i = 0; i < n; ++i) {
    _reactors.emplace_back(std::make_unique<MultiPollReactor>(this, per_reactor_max, shared));
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

bool MultiReactorPool::has_queued_work() const noexcept
{
  return _queue_size_hint.load(std::memory_order_relaxed) > 0;
}

std::size_t MultiReactorPool::queued_count_hint() const noexcept
{
  return _queue_size_hint.load(std::memory_order_relaxed);
}

std::size_t MultiReactorPool::fair_take_count() const noexcept
{
  auto const queued = _queue_size_hint.load(std::memory_order_relaxed);
  if (queued == 0) { return 0; }
  return std::max<std::size_t>((queued + _reactor_count - 1) / _reactor_count, 1);
}

bool MultiReactorPool::uses_first_available() const noexcept
{
  return _dispatch == RemoteReactorDispatch::FIRST_AVAILABLE;
}

std::unique_ptr<RemoteMultiTransfer> MultiReactorPool::try_pop_queued() noexcept
{
  std::lock_guard<std::mutex> const lock(_queue_mutex);
  if (_queue.empty()) { return nullptr; }
  auto transfer = std::move(_queue.front());
  _queue.pop_front();
  _queue_size_hint.store(_queue.size(), std::memory_order_relaxed);
  return transfer;
}

void MultiReactorPool::wake_reactors(std::size_t count) noexcept
{
  auto const n      = std::min(count, _reactor_count);
  auto const origin = _next_reactor_counter.fetch_add(n, std::memory_order_relaxed);
  for (std::size_t i = 0; i < n; ++i) {
    _reactors[(origin + i) % _reactor_count]->wakeup();
  }
}

void MultiReactorPool::submit_pread(std::vector<std::unique_ptr<RemoteMultiTransfer>> transfers)
{
  auto const reactor_count = _reactor_count;

  if (_dispatch == RemoteReactorDispatch::FIRST_AVAILABLE) {
    std::size_t queued_after = 0;
    std::exception_ptr fail_reason;
    {
      std::lock_guard<std::mutex> const lock(_queue_mutex);
      if (is_dead()) {
        fail_reason = death_reason();
      } else {
        for (auto& transfer : transfers) {
          _queue.push_back(std::move(transfer));
        }
        queued_after = _queue.size();
        _queue_size_hint.store(queued_after, std::memory_order_relaxed);
      }
    }
    if (fail_reason) {
      for (auto& transfer : transfers) {
        transfer->aggregate->on_subrange_failed(fail_reason);
      }
      return;
    }
    wake_reactors(queued_after);
    return;
  }

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

  // Drain the pool-wide queue.
  {
    std::deque<std::unique_ptr<RemoteMultiTransfer>> queued;
    {
      std::lock_guard<std::mutex> const lock(_queue_mutex);
      std::swap(queued, _queue);
      _queue_size_hint.store(0, std::memory_order_relaxed);
    }
    for (auto& transfer : queued) {
      transfer->aggregate->on_subrange_failed(eptr);
    }
  }

  // Wake every reactor out of curl_multi_poll so they notice _dead promptly. Including the caller's
  // own reactor is harmless, since it has already left its loop.
  for (auto const& r : _reactors) {
    r->wakeup();
  }
}

}  // namespace kvikio::detail
