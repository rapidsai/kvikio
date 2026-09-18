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
#include <vector>

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
    std::lock_guard const lock(_exception_mutex);
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
    std::lock_guard const lock(_exception_mutex);
    if (!_first_exception) { _first_exception = eptr; }
  }
  // Last thread to decrement to zero fulfills the promise.
  if (_subranges_left.fetch_sub(1, std::memory_order_acq_rel) == 1) {
    std::lock_guard const lock(_exception_mutex);
    if (recorder) { recorder->finish_with_failure(); }
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
    std::lock_guard const lock(_submit_mutex);
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

void MultiPollReactor::AdmitOutcome::set_ready_at(
  std::chrono::steady_clock::time_point ready_at) noexcept
{
  earliest_ready_at =
    earliest_ready_at.has_value() ? std::min(earliest_ready_at.value(), ready_at) : ready_at;
}

void MultiPollReactor::ingest_inbox()
{
  // The inbox is shared with submitting threads. Splice it out and drop the lock quickly.
  std::lock_guard const lock(_submit_mutex);
  if (_pending.empty()) {
    std::swap(_pending, _inbox);
    return;
  }
  while (!_inbox.empty()) {
    _pending.push_back(std::move(_inbox.front()));
    _inbox.pop_front();
  }
}

// Scratch state of one admission pass. `outcome` is what the pass hands back; the rest is only
// meaningful while the pass runs, which is why the type lives here rather than in the header.
struct MultiPollReactor::AdmitPass {
  AdmitOutcome outcome;

  // Taken once at the start of the pass. Backoffs are compared against it.
  std::chrono::steady_clock::time_point started_at{std::chrono::steady_clock::now()};

  // Contexts whose bounce-buffer shard already missed this pass. Distinct contexts are assumed
  // few, so a flat vector with linear find suffices.
  std::vector<CUcontext> exhausted_ctxs;
};

bool MultiPollReactor::try_admit(std::unique_ptr<RemoteMultiTransfer>& transfer, AdmitPass& pass)
{
  using BounceBufferCache = BounceBufferCachePerThreadAndContext<CudaPinnedAllocator>;

  // Still serving its retry backoff.
  if (transfer->ready_at > pass.started_at) {
    pass.outcome.set_ready_at(transfer->ready_at);
    return false;
  }

  // Gate 2 already missed for this context during this pass. Skip it without touching the limiter.
  // At worst this is pessimistic by one pass if a recycle frees a buffer mid-pass.
  auto const& exhausted = pass.exhausted_ctxs;
  if (transfer->is_device &&
      std::find(exhausted.begin(), exhausted.end(), transfer->device_ctx) != exhausted.end()) {
    pass.outcome.deferred_for_resource = true;
    return false;
  }

  // Gate 1 caps network concurrency: the HTTP range requests attached to this reactor's multi
  // handle at once, host and device combined. Use the slot the transfer arrived with (popped off
  // the pool-wide queue), else ask the limiter for one.
  auto slot = transfer->slot ? std::move(transfer->slot) : _request_limiter.try_acquire();
  if (!slot) {
    pass.outcome.deferred_for_resource = true;
    return false;
  }

  // Gate 2 caps bounce-buffer use per (reactor thread, CUDA context) across all pipeline phases.
  // A limiter slot is freed at libcurl completion, but the buffer stays in flight until the H2D
  // drains and the recycle callback fires. A refused transfer holds no slot while it waits: `slot`
  // goes out of scope here and returns to the limiter.
  if (transfer->is_device) {
    std::optional<CudaPinnedBounceBufferPool::Buffer> bounce_buffer;
    {
      PushAndPopContext c(transfer->device_ctx);
      bounce_buffer = BounceBufferCache::instance().try_get(transfer->device_ctx);
    }
    if (!bounce_buffer.has_value()) {
      pass.outcome.deferred_for_resource = true;
      pass.exhausted_ctxs.push_back(transfer->device_ctx);
      return false;
    }
    transfer->buffer            = std::move(bounce_buffer.value());
    transfer->ctx.pinned_buffer = transfer->buffer.get();
  }

  // Hand the easy handle to libcurl. A failure here is fatal for the pool. The transfer stays where
  // it is, so `fail_all_pending()` resolves it, along with everything else, with this exception.
  CURL* easy    = transfer->curl->handle();
  auto const mc = curl_multi_add_handle(_curl_multi, easy);
  KVIKIO_EXPECT(mc == CURLM_OK,
                std::string("curl_multi_add_handle: ") + curl_multi_strerror(mc),
                std::runtime_error);
  transfer->attachment = CurlMultiAttachment{_curl_multi, easy};
  transfer->slot       = std::move(slot);
  // The request is on the wire from here, so this is where the transfer's own span starts.
  // Everything before it was queueing, in the inbox or behind the gates.
  transfer->physical_recorder.emplace(
    transfer->physical, transfer->file_offset, transfer->ctx.size);
  _in_flight.emplace(easy, std::move(transfer));
  return true;
}

void MultiPollReactor::admit_from_pool(AdmitPass& pass)
{
  // Pool work comes after local work, so retries and carried-over transfers get slots first. The
  // share stops one reactor from running the tail of a burst alone.
  auto const share = _pool->queue_share_per_reactor();
  for (std::size_t taken = 0; taken < share; ++taken) {
    // Reserve before popping, so a sub-range leaves the queue only when a reactor can put it on
    // the wire. If the queue turns out empty, the slot returns to the limiter with `slot`.
    auto slot = _request_limiter.try_acquire();
    if (!slot) {
      pass.outcome.deferred_for_resource = true;
      return;
    }
    auto transfer = _pool->try_pop_queued();
    if (!transfer) { return; }
    transfer->slot = std::move(slot);

    bool admitted = false;
    try {
      admitted = try_admit(transfer, pass);
    } catch (...) {
      // Keep the transfer reachable, so `fail_all_pending()` resolves its aggregate.
      _pending.push_back(std::move(transfer));
      throw;
    }
    if (!admitted) {
      // Refused a bounce buffer. Pool work never waits in `_pending`, where no other reactor could
      // reach it. It goes back to the head of the queue, without a reservation, for whichever
      // reactor can start it.
      transfer->slot.reset();
      _pool->return_to_queue(std::move(transfer));
      return;
    }
  }
}

MultiPollReactor::AdmitOutcome MultiPollReactor::admit_pending()
{
  AdmitPass pass;
  // Local work first: retries and transfers carried over from earlier passes. An admitted transfer
  // has moved into `_in_flight`; a refused one stays in place, holding no slot.
  for (auto it = _pending.begin(); it != _pending.end();) {
    if (try_admit(*it, pass)) {
      it = _pending.erase(it);
    } else {
      ++it;
    }
  }
  if (_pool->uses_shared_queue()) { admit_from_pool(pass); }
  return pass.outcome;
}

void MultiPollReactor::perform()
{
  int running_handles = 0;
  auto const mc       = curl_multi_perform(_curl_multi, &running_handles);
  KVIKIO_EXPECT(mc == CURLM_OK,
                std::string("curl_multi_perform: ") + curl_multi_strerror(mc),
                std::runtime_error);
}

void MultiPollReactor::stage_device_copy(RemoteMultiTransfer& transfer)
{
  using BounceBufferCache = BounceBufferCachePerThreadAndContext<CudaPinnedAllocator>;

  // Phase A (network -> pinned) is done. Queue Phase B (pinned -> device) on this (thread, ctx)
  // stream and hand the buffer to a cuLaunchHostFunc recycle callback, so its cache slot returns
  // when the H2D drains. The callback also wakes this reactor, which may be waiting on that slot.
  PushAndPopContext c(transfer.device_ctx);
  CUstream stream = StreamCachePerThreadAndContext::get();
  KVIKIO_CUDA_DRIVER_TRY(cudaAPI::instance().MemcpyHtoDAsync(
    convert_void2deviceptr(transfer.device_dst), transfer.buffer.get(), transfer.ctx.size, stream));
  transfer.aggregate->io_event_barrier->record_event(stream);
  BounceBufferCache::instance().recycle_after(
    transfer.device_ctx, std::move(transfer.buffer), stream, [curl_multi = _curl_multi]() noexcept {
      std::ignore = curl_multi_wakeup(curl_multi);
    });
}

void MultiPollReactor::complete_transfer(std::unique_ptr<RemoteMultiTransfer> transfer,
                                         CURLcode result,
                                         AdmitOutcome& outcome)
{
  std::exception_ptr error;
  try {
    if (result == CURLE_OK && !transfer->ctx.overflow_error) {
      if (transfer->is_device) { stage_device_copy(*transfer); }
      // Before the aggregate, which may make the caller's future ready.
      transfer->physical_recorder->finish(transfer->ctx.size);
      transfer->aggregate->on_subrange_complete(transfer->ctx.size);
      return;
    }

    if (transfer->ctx.overflow_error) {
      // Prefer the handle's recorded error buffer. Fall back to the generic strerror text when
      // libcurl recorded no message.
      auto const errmsg = transfer->curl->error_message();
      error             = std::make_exception_ptr(std::runtime_error(
        std::string("curl_multi transfer failed (") +
        (errmsg.empty() ? std::string{curl_easy_strerror(result)} : errmsg) +
        ") [server returned more bytes than requested; maybe range support missing?]"));
    } else {
      long http_code = 0;
      transfer->curl->getinfo(CURLINFO_RESPONSE_CODE, &http_code);
      ++transfer->attempt;
      auto const errmsg  = transfer->curl->error_message();
      auto const verdict = transfer->retry_policy->evaluate(
        result, http_code, transfer->attempt, errmsg, "curl_multi transfer failed");
      if (verdict.decision == RetryDecision::RETRY) {
        KVIKIO_LOG_WARN(verdict.message);
        count_http_retry(verdict.delay_ms);
        auto const ready_at = std::chrono::steady_clock::now() + verdict.delay_ms;
        outcome.set_ready_at(ready_at);
        // Ends the failed attempt. The next admission starts a new observation, so the backoff
        // shows as a gap rather than as one long transfer.
        transfer->physical_recorder.reset();
        requeue_for_retry(std::move(transfer), ready_at);
        return;
      }
      error = std::make_exception_ptr(std::runtime_error(verdict.message));
    }
  } catch (...) {
    error = std::current_exception();
  }
  transfer->physical_recorder.reset();
  transfer->aggregate->on_subrange_failed(error);
}

std::size_t MultiPollReactor::reap_completions(AdmitOutcome& outcome)
{
  std::size_t completed = 0;
  int msgs_left         = 0;
  while (auto* msg = curl_multi_info_read(_curl_multi, &msgs_left)) {
    if (msg->msg != CURLMSG_DONE) { continue; }
    ++completed;
    auto* easy = msg->easy_handle;
    auto it    = _in_flight.find(easy);
    KVIKIO_EXPECT(it != _in_flight.end(),
                  "MultiPollReactor: completion for unknown handle",
                  std::runtime_error);
    auto transfer = std::move(it->second);
    _in_flight.erase(it);
    count_http_connection_of(easy);
    complete_transfer(std::move(transfer), msg->data.result, outcome);
  }
  return completed;
}

int MultiPollReactor::poll_timeout_ms(AdmitOutcome const& outcome,
                                      std::size_t completed) const noexcept
{
  // Nothing to admit. A submit, a completion, or a recycle callback wakes the poll early.
  constexpr int idle_timeout_ms = 1000;
  // Backstop while work waits on a slot or a bounce buffer. Both normally wake the poll on release,
  // by a completion or by the recycle callback.
  constexpr int busy_timeout_ms = 10;

  // Under SHARED_QUEUE an empty `_pending` is not idle while the pool-wide queue holds work this
  // reactor could still take. One refused a resource this pass cannot; its own completions wake it.
  bool const pool_work_waiting =
    _pool->uses_shared_queue() && !outcome.deferred_for_resource && _pool->queued_count_hint() > 0;
  if (_pending.empty() && !pool_work_waiting) { return idle_timeout_ms; }

  // Completions freed slots this pass. Come straight back and spend them on the waiting work.
  if (completed > 0) { return 0; }

  int timeout_ms = idle_timeout_ms;
  if (outcome.deferred_for_resource) { timeout_ms = busy_timeout_ms; }
  if (outcome.earliest_ready_at.has_value()) {
    // Wake for the earliest elapsed backoff, if that comes sooner.
    auto const wait_ms = std::chrono::ceil<std::chrono::milliseconds>(
                           outcome.earliest_ready_at.value() - std::chrono::steady_clock::now())
                           .count();
    timeout_ms = static_cast<int>(std::clamp<long long>(wait_ms, 0, timeout_ms));
  }
  return timeout_ms;
}

void MultiPollReactor::poll(int timeout_ms)
{
  auto const mc = curl_multi_poll(_curl_multi,
                                  nullptr,     // extra_fds
                                  0,           // extra_nfds
                                  timeout_ms,  // timeout_ms
                                  nullptr);    // numfds
  KVIKIO_EXPECT(
    mc == CURLM_OK, std::string("curl_multi_poll: ") + curl_multi_strerror(mc), std::runtime_error);
}

void MultiPollReactor::io_thread_main()
{
  // One pass: ingest -> admit -> perform -> reap -> poll. Two invariants keep the pass simple:
  //  - A transfer waiting in `_pending` holds no limiter slot. Slots are taken at admission and
  //    returned at completion (or, for a pool pop that is then refused, right away).
  //  - Pool-wide work never waits in `_pending`. A reactor takes it only when it can start it, and
  //    hands it back otherwise, so it is always reachable by whichever reactor has capacity.
  // Slots freed by this pass's completions are spent on the next pass, which `poll_timeout_ms()`
  // makes immediate.
  try {
    while (!_pool->is_dead()) {
      ingest_inbox();
      auto outcome = admit_pending();
      perform();
      auto const completed = reap_completions(outcome);
      poll(poll_timeout_ms(outcome, completed));
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
    std::lock_guard const lock(_submit_mutex);
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
    in_flight_entry.second->physical_recorder.reset();
    in_flight_entry.second->aggregate->on_subrange_failed(eptr);
  }
  _in_flight.clear();
}

namespace {
std::atomic<bool> _pool_instantiated{false};
}  // namespace

bool MultiReactorPool::is_instantiated() noexcept
{
  return _pool_instantiated.load(std::memory_order_acquire);
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

  // With no budget a reactor is never full, so nothing paces its takes from the queue.
  if (_dispatch == RemoteReactorDispatch::SHARED_QUEUE && max_total == 0) {
    KVIKIO_LOG_WARN(
      "KVIKIO_REMOTE_IO_REACTOR_DISPATCH=shared_queue needs a non-zero "
      "KVIKIO_REMOTE_IO_MAX_CONCURRENT_REQUESTS to pace the queue. Falling back to per_chunk.");
    _dispatch = RemoteReactorDispatch::PER_CHUNK;
  }

  // Slice the budget evenly, spreading any remainder one slot each over the first reactors so
  // the total is exact. A budget below the reactor count still gives every reactor one slot.
  auto const base      = max_total / n;
  auto const remainder = max_total % n;
  _reactors.reserve(n);
  for (std::size_t i = 0; i < n; ++i) {
    std::optional<std::size_t> const slice =
      (max_total == 0) ? std::nullopt
                       : std::optional{std::max<std::size_t>(base + (i < remainder ? 1 : 0), 1)};
    _reactors.emplace_back(std::make_unique<MultiPollReactor>(this, slice));
  }

  _pool_instantiated.store(true, std::memory_order_release);
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

std::size_t MultiReactorPool::queued_count_hint() const noexcept
{
  return _queue_size_hint.load(std::memory_order_relaxed);
}

std::size_t MultiReactorPool::queue_share_per_reactor() const noexcept
{
  auto const queued = _queue_size_hint.load(std::memory_order_relaxed);
  if (queued == 0) { return 0; }
  return std::max<std::size_t>((queued + _reactor_count - 1) / _reactor_count, 1);
}

bool MultiReactorPool::uses_shared_queue() const noexcept
{
  return _dispatch == RemoteReactorDispatch::SHARED_QUEUE;
}

std::unique_ptr<RemoteMultiTransfer> MultiReactorPool::try_pop_queued() noexcept
{
  std::lock_guard const lock(_queue_mutex);
  if (_queue.empty()) { return nullptr; }
  auto transfer = std::move(_queue.front());
  _queue.pop_front();
  _queue_size_hint.store(_queue.size(), std::memory_order_relaxed);
  return transfer;
}

void MultiReactorPool::return_to_queue(std::unique_ptr<RemoteMultiTransfer> transfer) noexcept
{
  std::exception_ptr fail_reason;
  {
    std::lock_guard const lock(_queue_mutex);
    if (is_dead()) {
      // `signal_death()` has already drained the queue. Nothing would ever pick this up again.
      fail_reason = death_reason();
    } else {
      try {
        // Head of the queue: it was next in line when the reactor took it.
        _queue.push_front(std::move(transfer));
        _queue_size_hint.store(_queue.size(), std::memory_order_relaxed);
        return;
      } catch (...) {
        // `push_front` failed before moving from `transfer`, which is still ours to fail.
        fail_reason = std::current_exception();
      }
    }
  }
  transfer->aggregate->on_subrange_failed(fail_reason);
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

  if (_dispatch == RemoteReactorDispatch::SHARED_QUEUE) {
    std::size_t queued_after = 0;
    std::exception_ptr fail_reason;
    {
      std::lock_guard const lock(_queue_mutex);
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
  std::lock_guard const lock(_death_mutex);
  return _death_reason;
}

void MultiReactorPool::signal_death(std::exception_ptr eptr) noexcept
{
  // The lock serializes _death_reason writes and keeps the _dead store in its scope so the first
  // writer wins, not the last. The store is `release`, pairing with the `acquire` in `is_dead()`.
  // The guard load below can be relaxed.
  {
    std::lock_guard const lock(_death_mutex);
    // Only the first thread here updates _death_reason and wakes reactors. Later calls early-exit.
    if (_dead.load(std::memory_order_relaxed)) { return; }
    _death_reason = eptr;
    _dead.store(true, std::memory_order_release);
  }

  // Drain the pool-wide queue.
  {
    std::deque<std::unique_ptr<RemoteMultiTransfer>> queued;
    {
      std::lock_guard const lock(_queue_mutex);
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
