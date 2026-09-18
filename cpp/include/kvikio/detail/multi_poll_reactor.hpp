/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <atomic>
#include <chrono>
#include <cstddef>
#include <deque>
#include <exception>
#include <future>
#include <memory>
#include <mutex>
#include <optional>
#include <thread>
#include <unordered_map>
#include <vector>

#include <curl/curl.h>

#include <kvikio/bounce_buffer.hpp>
#include <kvikio/detail/concurrent_request_limiter.hpp>
#include <kvikio/detail/http_retry.hpp>
#include <kvikio/detail/io_event_barrier.hpp>
#include <kvikio/detail/observation_recorder.hpp>
#include <kvikio/detail/remote_callback.hpp>
#include <kvikio/remote_handle.hpp>
#include <kvikio/shim/cuda.hpp>
#include <kvikio/shim/libcurl.hpp>

namespace kvikio::detail {

class MultiReactorPool;  // Forward declaration, because reactors needs to hold a back-pointer to
                         // the pool.

/**
 * @brief Given the max concurrent request cap for a reactor, derive the size of the libcurl
 * connection cache (`CURLMOPT_MAXCONNECTS`).
 *
 * @param max_concurrent_requests This reactor's private share of the total concurrent-request
 * budget (the global cap divided across reactors). `std::nullopt` means unlimited.
 * @return The value to pass to `CURLMOPT_MAXCONNECTS`. `std::nullopt` if @p max_concurrent_requests
 * is `std::nullopt` (unlimited concurrency).
 */
[[nodiscard]] std::optional<long> connection_cache_size(
  std::optional<std::size_t> max_concurrent_requests) noexcept;

/**
 * @brief Collects results from N sub-range transfers and resolves one top-level future once all of
 * them have either succeeded or one has failed.
 *
 * Every sub-range transfer belonging to a single `RemoteHandle::pread()` call holds a
 * `std::shared_ptr<RemoteMultiAggregateContext>`. As completions arrive on the reactor threads
 * (potentially in parallel when `KVIKIO_REMOTE_IO_NUM_REACTORS > 1`), each one calls
 * `on_subrange_complete()` or `on_subrange_failed()`. The thread that decrements `_subranges_left`
 * to zero fulfills `_promise`, with the accumulated byte total on success, or with the first
 * captured exception on failure.
 */
class RemoteMultiAggregateContext {
 public:
  /**
   * @brief Construct an aggregate that expects exactly `num_subranges` completion events.
   *
   * @param num_subranges Number of sub-range transfers the caller has split the read into.
   */
  explicit RemoteMultiAggregateContext(std::size_t num_subranges);

  /**
   * @brief Per-pread event barrier for the device-buffer path.
   */
  std::shared_ptr<IoEventBarrier> io_event_barrier;

  /**
   * @brief Records the logical operation these sub-ranges make up. Null when nobody is observing.
   */
  std::shared_ptr<LogicalObservationRecorder> recorder;

  /**
   * @brief Report that one sub-range transfer succeeded.
   *
   * @param bytes Number of bytes the sub-range delivered.
   */
  void on_subrange_complete(std::size_t bytes);

  /**
   * @brief Report that one sub-range transfer failed. The first exception captured wins.
   *
   * @param eptr The exception describing the failure.
   */
  void on_subrange_failed(std::exception_ptr eptr);

  /**
   * @brief Obtain the future the caller will observe. Must be called exactly once, before any
   * sub-range is submitted to the pool.
   */
  std::future<std::size_t> get_future();

 private:
  std::atomic<std::size_t> _subranges_left;
  std::atomic<std::size_t> _total_bytes{0};
  std::mutex _exception_mutex;
  std::exception_ptr _first_exception;
  std::promise<std::size_t> _promise;
};

/**
 * @brief RAII guard that keeps one libcurl easy handle attached to a multi handle.
 *
 * Set by the reactor right after a successful `curl_multi_add_handle`. Its destructor calls
 * `curl_multi_remove_handle`, so the handle is detached when the owning `RemoteMultiTransfer` is
 * destroyed. A default-constructed or moved-from guard is unset and does nothing on destruction.
 *
 * @note Must be destroyed on the reactor I/O thread that set it, because `CURLM*` is not
 * thread-safe. It is a `RemoteMultiTransfer` member declared after `curl`, so it detaches the
 * handle before `CurlHandle` returns it to the LibCurl pool.
 */
class CurlMultiAttachment {
 public:
  /**
   * @brief Construct an unset guard that holds no attachment.
   */
  CurlMultiAttachment() noexcept = default;

  /**
   * @brief Set a guard for an easy handle already attached to `multi`.
   *
   * @param multi The multi handle the easy handle was added to.
   * @param easy The easy handle to remove on destruction.
   */
  CurlMultiAttachment(CURLM* multi, CURL* easy) noexcept;

  ~CurlMultiAttachment();

  /**
   * @brief Explicitly detach the easy handle now instead of at destruction.
   */
  void reset() noexcept;

  // Move-only.
  CurlMultiAttachment(CurlMultiAttachment&& o) noexcept;
  CurlMultiAttachment& operator=(CurlMultiAttachment&& o) noexcept;
  CurlMultiAttachment(CurlMultiAttachment const&)            = delete;
  CurlMultiAttachment& operator=(CurlMultiAttachment const&) = delete;

 private:
  CURLM* _multi{nullptr};
  CURL* _easy{nullptr};
};

/**
 * @brief Per-transfer state owned by a `MultiPollReactor` between submission and completion.
 *
 * One `RemoteMultiTransfer` corresponds to one libcurl easy handle, which corresponds to one HTTP
 * range request. Sub-ranges of the same `pread()` share the same `aggregate`. The `curl` member is
 * held by `std::unique_ptr` because `CurlHandle` is intentionally non-movable.
 */
struct RemoteMultiTransfer {
  std::unique_ptr<CurlHandle> curl;

  // Detaches `curl`'s easy handle from the multi handle on destruction.
  CurlMultiAttachment attachment;

  CallbackContext ctx;
  std::shared_ptr<RemoteMultiAggregateContext> aggregate;

  // Concurrency slot, held from admission until this transfer is destroyed, which returns it. Also
  // held briefly between leaving the pool-wide queue and admission under SHARED_QUEUE. A
  // transfer waiting in a reactor's `_pending` never holds one.
  ConcurrentRequestLimiter::Slot slot;

  // Device-path fields. All zeroed/null for host transfers.
  bool is_device{false};
  CUcontext device_ctx{nullptr};
  void* device_dst{nullptr};
  CudaPinnedBounceBufferPool::Buffer buffer{nullptr, nullptr, 0};

  // Retry bookkeeping. Number of attempts that have finished.
  std::size_t attempt{0};

  // Byte offset of this sub-range in the remote object.
  std::size_t file_offset{0};

  // What every attempt of this sub-range shares.
  PhysicalObservationContext physical{};

  // The attempt currently on the wire. Started when the easy handle joins the multi handle,
  // finished at completion, and destroyed on failure or before a retry, so one attempt is one
  // observation and a backoff is a gap between two of them.
  std::optional<PhysicalObservationRecorder> physical_recorder;

  // Earliest time this transfer may be admitted. Used to space out retries.
  // The default is the clock epoch, which is always in the past, so a freshly submitted transfer is
  // admitted immediately.
  std::chrono::steady_clock::time_point ready_at{};

  std::shared_ptr<HttpRetryPolicy const> retry_policy;

  /**
   * @brief Recycles `buffer` to the bounce-buffer cache if it was not already moved out (due to
   * failure paths).
   **/
  ~RemoteMultiTransfer();
};

/**
 * @brief One reactor has one `CURLM*`, one I/O thread, one submit queue, one in-flight map.
 *
 * `CURLM*` is not thread-safe. All multi-side calls (`curl_multi_add_handle`, `curl_multi_perform`,
 * `curl_multi_info_read`, `curl_multi_remove_handle`, `curl_multi_poll`) happen on `_io_thread`.
 * The only cross-thread libcurl call is `curl_multi_wakeup()`, used by `submit()` to nudge the
 * reactor out of its poll.
 *
 * @note Instances are intentionally never destroyed. They are owned by the leaked
 * `MultiReactorPool` singleton, so their dtor body is empty. Reactor threads run until the process
 * exits.
 */
class MultiPollReactor {
 public:
  /**
   * @brief Construct a reactor owned by the given pool.
   *
   * @param pool Non-owning back-pointer to the pool that owns this reactor. Used to observe and
   * propagate pool-wide death state. The pool must outlive the reactor, which is guaranteed because
   * the pool is a leaked singleton that owns this reactor by `unique_ptr`.
   * @param max_concurrent_requests This reactor's private share of the total concurrent-request
   * budget (the global cap divided across reactors). `std::nullopt` means unlimited. Each reactor
   * enforces its own share.
   */
  MultiPollReactor(MultiReactorPool* pool, std::optional<std::size_t> max_concurrent_requests);
  ~MultiPollReactor() noexcept;
  MultiPollReactor(MultiPollReactor const&)            = delete;
  MultiPollReactor& operator=(MultiPollReactor const&) = delete;
  MultiPollReactor(MultiPollReactor&&)                 = delete;
  MultiPollReactor& operator=(MultiPollReactor&&)      = delete;

  /**
   * @brief Hand off a batch of prepared transfers to this reactor. Thread-safe.
   *
   * The reactor picks the transfers up on its next loop iteration. The caller must have already
   * obtained the aggregate future via `aggregate->get_future()` before calling this, because once
   * the transfers are in the queue the reactor may complete them (and the promise) at any time. If
   * the pool has already declared death, every transfer in the batch is failed immediately with
   * the recorded death reason and never enters the inbox.
   *
   * @param transfers Per-transfer state, ownership transferred to the reactor.
   */
  void submit(std::vector<std::unique_ptr<RemoteMultiTransfer>> transfers);

  /**
   * @brief Wake up the reactor out of its `curl_multi_poll()` wait. Thread-safe.
   *
   * This method calls `curl_multi_wakeup()`. If it fails (which is rare) the reactor still wakes on
   * its bounded poll timeout. Used by `MultiReactorPool::signal_death` to make every reactor notice
   * pool death promptly rather than waiting for the timeout.
   */
  void wakeup() noexcept;

 private:
  /**
   * @brief Set this reactor's libcurl connection cache (`CURLMOPT_MAXCONNECTS`).
   *
   * By default libcurl sets `CURLMOPT_MAXCONNECTS` to 4 x the number of easy handles attached to a
   * multi handle. This is recomputed on every transition, and a transient dip in concurrency will
   * cause libcurl to evict warm, reusable connections, and cause unnecessary TCP/TLS handshake.
   * Here we pin `CURLMOPT_MAXCONNECTS` to a fixed size.
   *
   * @param max_concurrent_requests This reactor's private share of the total concurrent-request
   * budget (the global cap divided across reactors). `std::nullopt` means unlimited.
   *
   * @exception std::runtime_error if `curl_multi_setopt` fails.
   */
  void set_connection_cache_size(std::optional<std::size_t> max_concurrent_requests) const;

  /**
   * @brief What one pass left behind: admission deferrals and retry backoffs. Decides the poll
   * timeout.
   */
  struct PassOutcome {
    // Earliest retry-backoff deadline among the transfers held back for one (if any).
    std::optional<std::chrono::steady_clock::time_point> earliest_ready_at;

    // Whether anything is held back for a limiter slot or a bounce buffer rather than for an
    // unelapsed retry backoff. The two take different poll timeouts.
    bool deferred_for_resource{false};

    /**
     * @brief Record a backoff deadline, keeping the earliest.
     *
     * @param ready_at The deadline.
     */
    void record_ready_at(std::chrono::steady_clock::time_point ready_at) noexcept;
  };

  // Scratch state of one admission pass, shared by every `try_admit()` call in it. Defined in the
  // .cpp next to its only users.
  struct AdmitPass;

  /**
   * @brief Splice newly submitted transfers out of the inbox into `_pending`.
   */
  void ingest_inbox();

  /**
   * @brief One admission pass: hand as many transfers to libcurl as the gates allow.
   *
   * Walks `_pending` first, so retries and transfers carried over from earlier passes get slots
   * before new work. Under `SHARED_QUEUE` it then pulls from the pool-wide queue. A transfer
   * that cannot be admitted stays in `_pending` if it is local, or goes back to the pool queue if
   * it came from there.
   *
   * @return What the pass left behind.
   */
  PassOutcome admit_pending();

  /**
   * @brief Run one transfer through the gates and, if it passes, attach it to the multi handle.
   *
   * @param transfer The transfer. Moved into `_in_flight` on success, left in place otherwise. A
   * refused transfer holds no limiter slot afterwards, even if it arrived with one.
   * @param pass The current pass's scratch state. Updated with why the transfer was refused.
   * @return Whether the transfer is now in flight.
   * @exception std::runtime_error if `curl_multi_add_handle` fails. The transfer is left in place
   * for `fail_all_pending()` to resolve.
   */
  bool try_admit(std::unique_ptr<RemoteMultiTransfer>& transfer, AdmitPass& pass);

  /**
   * @brief `SHARED_QUEUE` only. Pull sub-ranges off the pool-wide queue and admit them, up to
   * this reactor's share and while it has capacity.
   *
   * A sub-range leaves the queue only after a slot has been reserved for it, and goes straight back
   * if it is then refused a bounce buffer, so pool work never waits in `_pending`, where no other
   * reactor could reach it.
   *
   * @param pass The current pass's scratch state.
   */
  void admit_from_pool(AdmitPass& pass);

  /**
   * @brief One non-blocking `curl_multi_perform()`.
   *
   * @exception std::runtime_error on a libcurl multi-API error.
   */
  void perform();

  /**
   * @brief Drain libcurl's completion messages and settle each finished transfer.
   *
   * @param outcome Updated with the backoff deadline of any transfer requeued for retry.
   * @return How many transfers completed, successfully or not. Each has freed a limiter slot.
   */
  std::size_t reap_completions(PassOutcome& outcome);

  /**
   * @brief Settle one finished transfer: complete it, requeue it for retry, or fail it.
   *
   * @param transfer The transfer, already detached from `_in_flight`.
   * @param result libcurl's result code for the attempt.
   * @param outcome Updated with the backoff deadline if the transfer is requeued for retry.
   */
  void settle_transfer(std::unique_ptr<RemoteMultiTransfer> transfer,
                       CURLcode result,
                       PassOutcome& outcome);

  /**
   * @brief Queue the pinned-to-device copy of a finished device transfer and arrange for its
   * bounce buffer to return to the cache once that copy drains.
   *
   * @param transfer The finished device transfer. Its buffer is moved out.
   */
  void stage_device_copy(RemoteMultiTransfer& transfer);

  /**
   * @brief How long the next `curl_multi_poll()` may block, given what this pass left behind.
   *
   * @param outcome What admission left behind, plus any retry backoffs from completions.
   * @param completed How many transfers completed this pass.
   * @return The poll timeout in milliseconds. Zero when freed slots should be spent at once.
   */
  [[nodiscard]] int poll_timeout_ms(PassOutcome const& outcome,
                                    std::size_t completed) const noexcept;

  /**
   * @brief Block in `curl_multi_poll()` until socket activity, a wakeup, or the timeout.
   *
   * @param timeout_ms Longest time to block.
   * @exception std::runtime_error on a libcurl multi-API error.
   */
  void poll(int timeout_ms);

  void io_thread_main();

  /**
   * @brief Fail every transfer this reactor is responsible for and exit the loop.
   *
   * Called from the I/O thread on its way out, either because this reactor caught an exception or
   * because another reactor signaled pool death. Drains the inbox, removes each in-flight easy
   * handle from the multi handle, and resolves each transfer's aggregate with the given exception.
   */
  void fail_all_pending(std::exception_ptr eptr);

  /**
   * @brief Requeue a failed transfer in `_pending` so it can be attempted again.
   *
   * @param transfer The transfer to requeue. Ownership moves into `_pending`.
   * @param ready_at Earliest time the transfer may be admitted again.
   */
  void requeue_for_retry(std::unique_ptr<RemoteMultiTransfer> transfer,
                         std::chrono::steady_clock::time_point ready_at) noexcept;

  MultiReactorPool* _pool;
  ConcurrentRequestLimiter _request_limiter;
  CURLM* _curl_multi{nullptr};
  std::thread _io_thread;
  std::mutex _submit_mutex;
  std::deque<std::unique_ptr<RemoteMultiTransfer>> _inbox;
  std::deque<std::unique_ptr<RemoteMultiTransfer>> _pending;
  std::unordered_map<CURL*, std::unique_ptr<RemoteMultiTransfer>> _in_flight;
};

/**
 * @brief Process-wide pool that owns N reactors and dispatches sub-range transfers to them.
 *
 * Accessed via the leaked-pointer singleton `instance()`. Both `num_reactors` and the dispatch
 * mode are captured once at first use from `kvikio::defaults` and remain immutable for the process
 * lifetime: switching either requires restarting with different `KVIKIO_REMOTE_IO_NUM_REACTORS` /
 * `KVIKIO_REMOTE_IO_REACTOR_DISPATCH` env vars.
 *
 * Dispatch rules (with `N = _reactor_count`):
 *  - `PER_CHUNK` (default): each sub-range is routed independently via a round-robin atomic
 *    counter. Maximizes load distribution. May cause sub-ranges of the same file to use distinct
 *    TCP/TLS connections.
 *  - `PER_PREAD`: all sub-ranges of one `submit_pread()` call land on the same reactor (round-robin
 *    per call). Preserves per-`CURLM` connection-pool reuse.
 *  - `SHARED_QUEUE`: sub-ranges wait in one pool-wide queue. A reactor takes one only after
 *    reserving concurrency for it, and hands it back if it cannot start it at once, so work binds
 *    at execution time rather than submission time. Needs a non-zero concurrency budget to pace
 *    the queue.
 */
class MultiReactorPool {
 public:
  /**
   * @brief Get the process-wide pool, creating it (and its reactor threads) on first use.
   *
   * @note The returned reference points to a heap-allocated singleton that is intentionally never
   * destroyed, mirroring the leak convention used by `BounceBufferPool` and
   * `StreamCachePerThreadAndContext`. This avoids static-destruction-order coupling between the
   * pool, `LibCurl`, the reactor threads, and (future) CUDA teardown.
   */
  static MultiReactorPool& instance();

  /**
   * @brief Whether the pool singleton has already been constructed.
   *
   * `num_reactors`, the dispatch mode, and the concurrency cap are all captured once in the
   * pool's constructor, so changing them after this returns `true` would silently have no effect.
   * Used by `kvikio::defaults` to reject such changes with an exception instead.
   */
  [[nodiscard]] static bool is_instantiated() noexcept;

  MultiReactorPool(MultiReactorPool const&)            = delete;
  MultiReactorPool& operator=(MultiReactorPool const&) = delete;
  MultiReactorPool(MultiReactorPool&&)                 = delete;
  MultiReactorPool& operator=(MultiReactorPool&&)      = delete;

  /**
   * @brief Submit all sub-range transfers belonging to one `RemoteHandle::pread()` call.
   *
   * Routes each transfer to a reactor according to the captured dispatch policy. The caller must
   * have already obtained the aggregate future from the shared `RemoteMultiAggregateContext`
   * before invoking this, because as soon as the pool returns the reactors may have already
   * started completing the transfers.
   *
   * @param transfers The sub-range transfers, ownership transferred to the pool.
   */
  void submit_pread(std::vector<std::unique_ptr<RemoteMultiTransfer>> transfers);

  /**
   * @brief Whether the pool has been marked dead by a reactor that has caught a fatal libcurl
   * error.
   *
   * Once dead, the pool stays dead for the rest of the process lifetime. All in-flight and
   * subsequently submitted transfers fail with the recorded death reason.
   */
  [[nodiscard]] bool is_dead() const noexcept;

  /**
   * @brief Get the exception that caused pool death, or a null `exception_ptr` if alive.
   *
   * Safe to call from any thread. Returns the same value once `is_dead()` returns `true`.
   */
  [[nodiscard]] std::exception_ptr death_reason() const noexcept;

  /**
   * @brief Mark the pool as dead with the given exception as the cause, then wake every reactor so
   * each notices the death state promptly. Thread-safe. Only the first call wins. All subsequent
   * calls are silently ignored.
   *
   * @param eptr The exception that causes pool death. Will be propagated to every in-flight and
   * subsequently submitted transfer via `RemoteMultiAggregateContext::on_subrange_failed`.
   */
  void signal_death(std::exception_ptr eptr) noexcept;

  /**
   * @brief Take one sub-range off the pool-wide queue, or nothing if it is empty. Thread-safe.
   *
   * The caller must already hold a concurrency reservation for it.
   *
   * @return The sub-range, or null when the queue is empty.
   */
  [[nodiscard]] std::unique_ptr<RemoteMultiTransfer> try_pop_queued() noexcept;

  /**
   * @brief Put a sub-range back at the head of the pool-wide queue. Thread-safe.
   *
   * For a reactor that took the sub-range but cannot start it after all. If the pool has died in
   * the meantime the sub-range is failed with the death reason instead, since nothing would ever
   * drain the queue again.
   *
   * @param transfer The sub-range, without a concurrency reservation.
   */
  void return_to_queue(std::unique_ptr<RemoteMultiTransfer> transfer) noexcept;

  /**
   * @brief Roughly how many sub-ranges the pool-wide queue holds. A hint, see `_queue_size_hint`.
   * Thread-safe.
   */
  [[nodiscard]] std::size_t queued_count_hint() const noexcept;

  /**
   * @brief One reactor's even share of the pool-wide queue, as a number of sub-ranges.
   *
   * Derived from `queued_count_hint()`, so a hint as well.
   *
   * @return At least 1 whenever the queue is non-empty, so a reactor can always make progress.
   */
  [[nodiscard]] std::size_t queue_share_per_reactor() const noexcept;

  /**
   * @brief Whether sub-ranges are parked in the pool-wide queue instead of pushed to a reactor.
   */
  [[nodiscard]] bool uses_shared_queue() const noexcept;

  /**
   * @brief Nudge every reactor out of its poll. Thread-safe.
   *
   * Called on every shared-queue submit and on pool death. Waking only as many reactors as there
   * are sub-ranges would leave idle reactors asleep whenever the woken ones happen to be full, so a
   * small submit would wait for one of their completions or for the idle tick. N socketpair writes
   * per submit is cheap next to that.
   */
  void wake_all_reactors() noexcept;

 private:
  MultiReactorPool();
  ~MultiReactorPool() noexcept;

  // Fixed at construction. Reactor threads start while `_reactors` is still being filled, which
  // makes its `size()` unsafe for them to read.
  std::size_t _reactor_count;
  std::vector<std::unique_ptr<MultiPollReactor>> _reactors;
  RemoteReactorDispatch _dispatch;
  // Round-robin counter. Incremented per pread (PER_PREAD) or per chunk (PER_CHUNK).
  std::atomic<std::size_t> _next_reactor_counter{0};
  std::atomic<bool> _dead{false};
  std::mutex mutable _death_mutex;  // Protects writes to `_death_reason`.
  std::exception_ptr _death_reason;

  // SHARED_QUEUE only. Sub-ranges wait here until some reactor has a slot for one.
  std::mutex _queue_mutex;
  std::deque<std::unique_ptr<RemoteMultiTransfer>> _queue;
  // Mirrors `_queue.size()`. Written under `_queue_mutex`, read without it, so it is only a reason
  // to try `try_pop_queued()`, never proof that work is there.
  std::atomic<std::size_t> _queue_size_hint{0};
};

}  // namespace kvikio::detail
