/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <atomic>
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
#include <kvikio/detail/io_event_barrier.hpp>
#include <kvikio/detail/remote_callback.hpp>
#include <kvikio/remote_handle.hpp>
#include <kvikio/shim/cuda.hpp>
#include <kvikio/shim/libcurl.hpp>

namespace kvikio::detail {

class MultiReactorPool;  // Forward declaration, because reactors needs to hold a back-pointer to
                         // the pool.

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
   * @brief Optional per-pread event watermark for the device-buffer path.
   *
   * Populated by `RemoteHandle::pread` when the destination is device memory and shared by all
   * sub-range transfers belonging to this pred. The reactor records on it after each
   * `cuMemcpyAsync`; the caller's deferred future waits on `sync_all_events()` before returning.
   * Null for host transfers.
   */
  std::shared_ptr<IoEventBarrier> io_event_barrier;

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
 * Armed by the reactor immediately after a successful `curl_multi_add_handle`. Its destructor calls
 * `curl_multi_remove_handle`, so the easy handle is detached exactly when the owning
 * `RemoteMultiTransfer` is destroyed. Move-only; a default-constructed or moved-from guard is inert
 * and its destructor is a no-op.
 *
 * @note Must be destroyed on the reactor I/O thread that armed it. `CURLM*` is not thread-safe, so
 * every multi-side call, including this removal, must stay on that thread. This is the same
 * thread-affinity constraint the bounce buffer has.
 *
 * @note Held as a `RemoteMultiTransfer` member declared after `curl`, so it destructs before `curl`
 * does. That ordering guarantees the handle is removed from the multi handle before `CurlHandle`
 * returns it to the LibCurl free pool.
 */
class CurlMultiAttachment {
 public:
  /**
   * @brief Construct an inert guard that holds no attachment.
   */
  CurlMultiAttachment() noexcept = default;

  /**
   * @brief Arm a guard for an easy handle already attached to `multi`.
   *
   * @param multi The multi handle the easy handle was added to.
   * @param easy The easy handle to remove on destruction.
   */
  CurlMultiAttachment(CURLM* multi, CURL* easy) noexcept;

  ~CurlMultiAttachment();

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
 *
 * Device-buffer fields (`is_device`, `device_ctx`, `device_dst`, `buffer`) are populated by the
 * pred submitter when the destination is device memory and consumed by the reactor's stages (1)
 * and (3). For host transfers, `is_device` is false and the other device fields are unused.
 */
struct RemoteMultiTransfer {
  std::unique_ptr<CurlHandle> curl;

  // Keeps `curl`'s easy handle attached to the reactor's multi handle. Declared right after `curl`
  // so it destructs before `curl` returns the handle to the LibCurl pool: the handle is removed
  // from the multi handle first. Armed in reactor stage (1) after a successful
  // `curl_multi_add_handle`; inert until then.
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
  // Pinned bounce buffer checked out from the cache in reactor stage (1). The reactor moves this
  // into `cache.recycle_after` once the H2D is scheduled in stage (3). On the failure paths when
  // the buffer was not moved, the destructor recycles it via recycle_now.
  CudaPinnedBounceBufferPool::Buffer buffer{nullptr, nullptr, 0};

  // Recycles `buffer` to the bounce-buffer cache if it was not already moved out (failure paths).
  // Must run on the reactor I/O thread that checked the buffer out; see the definition in
  // multi_poll_reactor.cpp for the thread-affinity invariant.
  ~RemoteMultiTransfer();
};

/**
 * @brief One reactor has one `CURLM*`, one I/O thread, one submit queue, one in-flight map.
 *
 * `CURLM*` is not thread-safe; all multi-side calls (`curl_multi_add_handle`, `curl_multi_perform`,
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
   * enforces its own share against its own inbox.
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
  void io_thread_main();

  /**
   * @brief Fail every transfer this reactor is responsible for and exit the loop.
   *
   * Called from the I/O thread on its way out, either because this reactor caught an exception or
   * because another reactor signaled pool death. Drains the inbox, removes each in-flight easy
   * handle from the multi handle, and resolves each transfer's aggregate with the given exception.
   */
  void fail_all_pending(std::exception_ptr eptr);

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
 * Dispatch rules (with `N = _reactors.size()`):
 *  - `PER_CHUNK` (default): each sub-range is routed independently via a round-robin atomic
 *    counter. Maximizes load distribution; may cause sub-ranges of the same file to use distinct
 *    TCP/TLS connections.
 *  - `PER_PREAD`: all sub-ranges of one `submit_pread()` call land on the same reactor (round-robin
 *    per call). Preserves per-`CURLM` connection-pool reuse.
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

 private:
  MultiReactorPool();
  ~MultiReactorPool() noexcept;

  std::vector<std::unique_ptr<MultiPollReactor>> _reactors;
  RemoteReactorDispatch _dispatch;
  // Round-robin counter. Incremented per pread (PER_PREAD) or per chunk (PER_CHUNK).
  std::atomic<std::size_t> _next_reactor_counter{0};
  std::atomic<bool> _dead{false};
  std::mutex mutable _death_mutex;  // Protects writes to `_death_reason`.
  std::exception_ptr _death_reason;
};

}  // namespace kvikio::detail
