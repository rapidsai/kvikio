/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <atomic>
#include <condition_variable>
#include <future>
#include <optional>
#include <queue>
#include <thread>
#include <utility>

#include <kvikio/detail/function_wrapper.hpp>

/**
 * @file
 * @brief A simple, header-only thread pool that uses per-thread task queues. Synchronization only
 * exists between the pairs of the main thread and each worker thread, but not among the worker
 * threads themselves. Inspired by the BS threadpool that KvikIO has been using.
 */

namespace kvikio {
/**
 * @brief Utility class for the calling thread.
 */
class ThisThread {
 public:
  /**
   * @brief Check if the calling thread is from RoundRobinThreadPool.
   *
   * @return Boolean answer.
   */
  static bool is_from_pool() { return get_thread_idx().has_value(); }

  /**
   * @brief Get the index of the calling thread.
   *
   * If the calling thread is not from RoundRobinThreadPool, return std::nullopt. Otherwise, return
   * the thread index ranging from 0 to (N-1) where N is the thread count.
   *
   * @return Index of the calling thread.
   */
  static std::optional<std::size_t> get_thread_idx() { return this_thread_idx; }

 private:
  friend class RoundRobinThreadPool;

  /**
   * @brief Set the index of the calling thread.
   *
   * @param thread_idx Index of the calling thread.
   */
  static void set_thread_idx(std::size_t thread_idx) { this_thread_idx = thread_idx; }

  inline static thread_local std::optional<std::size_t> this_thread_idx{std::nullopt};
};

/**
 * @brief
 *
 */
struct Worker {
  std::thread thread;
  std::condition_variable task_available_cv;
  std::condition_variable task_done_cv;
  std::mutex task_mutex;
  std::queue<detail::FunctionWrapper> task_queue;
  std::size_t tasks_in_flight{0};
  bool should_stop{false};
};

/**
 * @brief A simple thread pool that uses per-thread task queues.
 *
 * Each worker thread has their own task queue, mutex and condition variable. The per-thread
 * synchronization primitives (mutex and condition variable) are shared with the main thread. Tasks
 * are submitted to the worker threads in a round-robin fashion, unless the target thread index is
 * specified by the user.
 *
 * Example:
 * @code{.cpp}
 * // Create a thread pool with 4 threads, and pass an optional callable with which to initialize
 * // each worker thread.
 * kvikio::RoundRobinThreadPool thread_pool{4, [] {
 *     // Initialize worker thread
 * }};
 *
 * // Submit the task to the thread pool. The worker thread is selected automatically in a
 * // round-robin fashion.
 * auto fut = thread_pool.submit_task([] {
 *     // Task logic
 * });
 *
 * // Submit the task to a specific thread.
 * auto fut = thread_pool.submit_task_to_thread([] {
 *     // Task logic
 * });
 *
 * // Wait until the result is ready.
 * auto result = fut.get();
 * @endcode
 */
class RoundRobinThreadPool {
 public:
  /**
   * @brief Constructor. Create a thread pool.
   *
   * @tparam F Type of the user-defined worker thread initialization.
   * @param num_threads Number of threads.
   * @param worker_thread_init_func User-defined worker thread initialization.
   */
  template <typename F>
  RoundRobinThreadPool(unsigned int num_threads, F&& worker_thread_init_func)
    : _num_threads{num_threads}, _worker_thread_init_func{std::forward<F>(worker_thread_init_func)}
  {
    create_threads();
  }

  /**
   * @brief Constructor, without user-defined worker thread initialization.
   *
   * @param num_threads Number of threads.
   */
  RoundRobinThreadPool(unsigned int num_threads)
    : RoundRobinThreadPool(num_threads, detail::FunctionWrapper{})
  {
  }

  /**
   * @brief Destructor. Wait until all worker threads complete their tasks, then join the threads.
   */
  ~RoundRobinThreadPool()
  {
    wait();
    destroy_threads();
  }

  /**
   * @brief Wait until all worker threads complete their tasks. Then join the threads, and
   * reinitialize the thread pool with new threads.
   *
   * @tparam F Type of the user-defined worker thread initialization.
   * @param num_threads Number of threads.
   * @param worker_thread_init_func User-defined worker thread initialization.
   */
  template <typename F>
  void reset(unsigned int num_threads, F&& worker_thread_init_func)
  {
    wait();
    destroy_threads();

    _num_threads             = num_threads;
    _worker_thread_init_func = std::forward<F>(worker_thread_init_func);
    create_threads();
  }

  /**
   * @brief Overload of reset(), without user-defined worker thread initialization.
   *
   * @param num_threads Number of threads.
   */
  void reset(unsigned int num_threads) { reset(num_threads, detail::FunctionWrapper{}); }

  /**
   * @brief Block the calling thread until all worker threads complete their tasks.
   */
  void wait()
  {
    for (unsigned int thread_idx = 0; thread_idx < _num_threads; ++thread_idx) {
      auto& worker = _workers[thread_idx];
      std::unique_lock lock(worker.task_mutex);
      worker.task_done_cv.wait(
        lock, [&] { return worker.task_queue.empty() && worker.tasks_in_flight == 0; });
    }
  }

  /**
   * @brief Get the number of threads from the thread pool.
   *
   * @return Thread count.
   */
  unsigned int num_threads() const { return _num_threads; }

  /**
   * @brief Submit the task to the thread pool for execution. The worker thread is selected
   * automatically in a round-robin fashion.
   *
   * @tparam F Type of the task callable.
   * @tparam R Return type of the task callable.
   * @param task  Task callable. The task can either be copyable or move-only.
   * @return An std::future<R> object. R can be void or other types.
   */
  template <typename F, typename R = std::invoke_result_t<std::decay_t<F>>>
  requires std::invocable<std::decay_t<F>> [[nodiscard]] std::future<R> submit_task(F&& task)
  {
    // The call index is atomically incremented on each submit_task call, and will wrap around once
    // it reaches the maximum value the integer type `std::size_t` can hold (this overflow
    // behavior is well-defined in C++).
    auto tid =
      std::atomic_fetch_add_explicit(&_task_submission_counter, 1, std::memory_order_relaxed);
    tid %= _num_threads;

    return submit_task_to_thread(std::forward<F>(task), tid);
  }

  /**
   * @brief Submit the task to a specific thread for execution.
   *
   * @tparam F Type of the task callable.
   * @tparam R Return type of the task callable.
   * @param task Task callable. The task can either be copyable or move-only.
   * @param thread_idx Index of the thread to which the task is submitted.
   * @return An std::future<R> object. R can be void or other types.
   */
  template <typename F, typename R = std::invoke_result_t<std::decay_t<F>>>
  [[nodiscard]] std::future<R> submit_task_to_thread(F&& task, std::size_t thread_idx)
  {
    auto& worker = _workers[thread_idx];

    std::promise<R> p;
    auto fut = p.get_future();

    {
      std::lock_guard lock(worker.task_mutex);

      worker.task_queue.emplace([task = std::forward<F>(task), p = std::move(p)]() mutable {
        try {
          if constexpr (std::is_same_v<R, void>) {
            task();
            p.set_value();
          } else {
            p.set_value(task());
          }
        } catch (...) {
          p.set_exception(std::current_exception());
        }
      });

      ++worker.tasks_in_flight;
    }

    worker.task_available_cv.notify_one();
    return fut;
  }

 private:
  /**
   * @brief Worker thread loop.
   *
   * @param thread_idx Worker thread index.
   */
  void run_worker(std::size_t thread_idx)
  {
    ThisThread::set_thread_idx(thread_idx);

    auto& worker = _workers[thread_idx];

    if (_worker_thread_init_func) { std::invoke(_worker_thread_init_func); }

    while (true) {
      std::unique_lock lock(worker.task_mutex);

      if (worker.task_queue.empty() && worker.tasks_in_flight == 0) {
        worker.task_done_cv.notify_all();
      }

      worker.task_available_cv.wait(
        lock, [&] { return !worker.task_queue.empty() || worker.should_stop; });

      if (worker.should_stop) { break; }

      auto task = std::move(worker.task_queue.front());
      worker.task_queue.pop();
      lock.unlock();

      task();

      {
        std::lock_guard done_lock(worker.task_mutex);
        --worker.tasks_in_flight;
      }
    }
  }

  /**
   * @brief Create worker threads.
   */
  void create_threads()
  {
    _workers = std::make_unique<Worker[]>(_num_threads);
    for (unsigned int thread_idx = 0; thread_idx < _num_threads; ++thread_idx) {
      _workers[thread_idx].thread = std::thread([this, thread_idx] { run_worker(thread_idx); });
    }
  }

  /**
   * @brief Notify each work thread of the intention to stop and join the threads. Pre-condition:
   * Each worker thread has finished all the tasks in their task queue.
   */
  void destroy_threads()
  {
    for (unsigned int thread_idx = 0; thread_idx < _num_threads; ++thread_idx) {
      auto& worker = _workers[thread_idx];

      {
        std::lock_guard lock(worker.task_mutex);
        worker.should_stop = true;
      }

      worker.task_available_cv.notify_one();

      worker.thread.join();
    }
  }

  unsigned int _num_threads{};
  detail::FunctionWrapper _worker_thread_init_func;
  std::unique_ptr<Worker[]> _workers;
  std::atomic_size_t _task_submission_counter{0};
};

}  // namespace kvikio
