/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
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
#include <thread>
#include <unordered_map>
#include <vector>

#include <curl/curl.h>

#include <kvikio/detail/remote_callback.hpp>
#include <kvikio/remote_handle.hpp>
#include <kvikio/shim/libcurl.hpp>

namespace kvikio::detail {

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
   * @brief Report that one sub-range transfer succeeded.
   *
   * @param bytes Number of bytes the sub-range delivered.
   */
  void on_subrange_complete(std::size_t bytes);

  /**
   * @brief Report that one sub-range transfer failed. The first exception captured wins.
   *
   * @param ep The exception describing the failure.
   */
  void on_subrange_failed(std::exception_ptr ep);

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
 * @brief Per-transfer state owned by a `MultiPollReactor` between submission and completion.
 *
 * One `RemoteMultiTransfer` corresponds to one libcurl easy handle, which corresponds to one HTTP
 * range request. Sub-ranges of the same `pread()` share the same `agg`. The `curl` member is held
 * by `std::unique_ptr` because `CurlHandle` is intentionally non-movable.
 */
struct RemoteMultiTransfer {
  std::unique_ptr<CurlHandle> curl;
  CallbackContext ctx;
  std::shared_ptr<RemoteMultiAggregateContext> aggregate;
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
  MultiPollReactor();
  ~MultiPollReactor() noexcept;
  MultiPollReactor(MultiPollReactor const&)            = delete;
  MultiPollReactor& operator=(MultiPollReactor const&) = delete;
  MultiPollReactor(MultiPollReactor&&)                 = delete;
  MultiPollReactor& operator=(MultiPollReactor&&)      = delete;

  /**
   * @brief Hand off a prepared transfer to this reactor. Thread-safe.
   *
   * The reactor picks the transfer up on its next loop iteration. The caller must have already
   * obtained the aggregate future via `agg->get_future()` before calling this, because once the
   * transfer is in the queue the reactor may complete it (and the promise) at any time.
   *
   * @param transfer Per-transfer state, ownership transferred to the reactor.
   */
  void submit(std::unique_ptr<RemoteMultiTransfer> transfer);

 private:
  void io_thread_main();

  CURLM* _curl_multi{nullptr};
  std::thread _io_thread;
  std::mutex _submit_mutex;
  std::deque<std::unique_ptr<RemoteMultiTransfer>> _inbox;
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

 private:
  MultiReactorPool();
  ~MultiReactorPool() noexcept;

  std::vector<std::unique_ptr<MultiPollReactor>> _reactors;
  RemoteReactorDispatch _dispatch;
  std::atomic<std::size_t> _per_pread_counter{0};  // Round-robin counter for PER_PREAD mode.
  std::atomic<std::size_t> _per_chunk_counter{0};  // Round-robin counter for PER_CHUNK mode.
};

}  // namespace kvikio::detail
