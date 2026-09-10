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
 * @brief Cap on bounce buffers held per (reactor thread, CUDA context).
 *
 * Tracks whichever regime the concurrency budget uses. A sliced budget lets a reactor start at
 * most its own share of device transfers, so a matching slice of buffers is enough. A pool-wide
 * budget lets one reactor hold far more than that share, and a sliced buffer cap would then stop
 * it from starting device work it has already reserved concurrency for, stranding that budget
 * where no other reactor can use it. Total pinned memory stays bounded either way, because every
 * in-flight device transfer needs a slot as well as a buffer.
 *
 * @return The cap. `std::nullopt` means unlimited.
 */
[[nodiscard]] std::optional<std::size_t> bounce_buffer_cap();

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

  // Concurrency slot held from stage (1) admission until this transfer is destroyed after
  // completion or failure. Empty while the transfer waits in the inbox. Destroying the transfer
  // returns the slot to the reactor's limiter.
  ConcurrentRequestLimiter::Slot slot;

  // Device-path fields. All zeroed/null for host transfers.
  bool is_device{false};
  CUcontext device_ctx{nullptr};
  void* device_dst{nullptr};
  CudaPinnedBounceBufferPool::Buffer buffer{nullptr, nullptr, 0};

  // Retry bookkeeping. Number of attempts that have finished.
  std::size_t attempt{0};

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
   * budget (the global cap divided across reactors). `std::nullopt` means unlimited. Only consulted
   * when `shared_limiter` is null.
   * @param shared_limiter Pool-wide limiter holding the whole concurrency budget, or null to give
   * this reactor a private limiter sized to `max_concurrent_requests`.
   */
  MultiPollReactor(MultiReactorPool* pool,
                   std::optional<std::size_t> max_concurrent_requests,
                   ConcurrentRequestLimiter* shared_limiter);
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
   * @brief What one admission walk over `_pending` left behind.
   */
  struct AdmitOutcome {
    // Earliest retry-backoff deadline among the deferred transfers (if any).
    std::optional<std::chrono::steady_clock::time_point> earliest_ready_at;

    // Whether anything is held back for a limiter slot or a bounce buffer rather than for an
    // unelapsed retry backoff. The two take different poll timeouts.
    bool deferred_for_resource{false};

    /**
     * @brief Merge a later walk's result into this one.
     *
     * @param other The later walk's outcome.
     */
    void merge(AdmitOutcome const& other) noexcept;
  };

  /**
   * @brief Hand as many pending transfers to libcurl as the gates currently allow.
   *
   * Transfers that cannot be admitted stay in `_pending` for a later walk.
   *
   * @return What the walk left behind. Decides this pass's poll timeout.
   */
  AdmitOutcome admit_pending();

  /**
   * @brief Move as much of the pool-wide queue into `_pending` as this reactor has capacity for.
   *
   * `FIRST_AVAILABLE` only. Each transfer taken carries the concurrency reservation that allowed it
   * to be taken, leaving `admit_pending()` free to hand it straight to libcurl.
   *
   * @param outcome Updated when the queue still holds work this reactor could not reserve for.
   */
  void take_from_pool_queue(AdmitOutcome& outcome);

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
  ConcurrentRequestLimiter _private_limiter;

  // Points at either `_private_limiter` or the pool's shared one. Never null.
  ConcurrentRequestLimiter* _request_limiter;

  // `FIRST_AVAILABLE` only. Most sub-ranges this reactor can accept.
  std::size_t _take_ceiling;
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
 *  - `FIRST_AVAILABLE`: sub-ranges are parked in one pool-wide queue. A reactor takes one only
 * after reserving concurrency for it, which binds work at execution time instead of submission
 * time. Needs a non-zero concurrency budget to pace the queue.
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
   * @brief Whether the pool-wide queue currently holds anything. Thread-safe.
   *
   * A hint, not a guarantee. The queue may be drained between the check and the next pop.
   */
  [[nodiscard]] bool has_queued_work() const noexcept;

  /**
   * @brief Roughly how many sub-ranges the pool-wide queue holds. Thread-safe.
   *
   * Fit for sizing a batch, as `fair_take_count()` and the wake path do, and not for anything
   * that needs the count to be right. Reads without the queue lock, so it can miss a push or a
   * pop that is already under way, and is stale as soon as it returns.
   */
  [[nodiscard]] std::size_t queued_count_hint() const noexcept;

  /**
   * @brief How many sub-ranges one reactor may take from the pool-wide queue right now.
   *
   * An even split of what is queued. Without it, whichever reactor asks first sweeps up the tail
   * of a burst and runs it alone, and a `pread()` waits on its slowest sub-range.
   *
   * @return At least 1 whenever the queue is non-empty, which keeps progress possible.
   */
  [[nodiscard]] std::size_t fair_take_count() const noexcept;

  /**
   * @brief Whether sub-ranges are parked in the pool-wide queue instead of pushed to a reactor.
   */
  [[nodiscard]] bool uses_first_available() const noexcept;

  /**
   * @brief Nudge `count` reactors, chosen round-robin, out of their poll. Thread-safe.
   *
   * Freeing a pool-wide slot lets some other reactor start queued work. Only a wakeup tells it.
   *
   * @param count How many reactors to wake. Clamped to the reactor count.
   */
  void wake_reactors(std::size_t count) noexcept;

 private:
  MultiReactorPool();
  ~MultiReactorPool() noexcept;

  // Fixed at construction. Reactor threads start while `_reactors` is still being filled, which
  // makes its `size()` unsafe for them to read.
  std::size_t _reactor_count;
  // Declared before `_reactors` to outlive the reactors that borrow it.
  std::optional<ConcurrentRequestLimiter> _shared_limiter;
  std::vector<std::unique_ptr<MultiPollReactor>> _reactors;
  RemoteReactorDispatch _dispatch;
  // Round-robin counter. Incremented per pread (PER_PREAD) or per chunk (PER_CHUNK), and used to
  // rotate which reactors `wake_reactors()` nudges.
  std::atomic<std::size_t> _next_reactor_counter{0};
  std::atomic<bool> _dead{false};
  std::mutex mutable _death_mutex;  // Protects writes to `_death_reason`.
  std::exception_ptr _death_reason;

  // FIRST_AVAILABLE only. Sub-ranges wait here until some reactor has a slot for one.
  std::mutex _queue_mutex;
  std::deque<std::unique_ptr<RemoteMultiTransfer>> _queue;
  // Mirrors `_queue.size()` so the reactors can ask whether taking is worth attempting without
  // serializing on `_queue_mutex` every loop pass. Written under that mutex, read without it, so a
  // reader can catch the gap between a push or pop and the store, and any value is stale the
  // moment it is returned. Only ever a reason to try. `try_pop_queued()` returning null is what
  // settles whether work was really there.
  std::atomic<std::size_t> _queue_size_hint{0};
};

}  // namespace kvikio::detail
