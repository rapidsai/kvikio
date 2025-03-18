/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions andc
 * limitations under the License.
 */

#include <atomic>
#include <condition_variable>
#include <future>
#include <optional>
#include <queue>
#include <thread>
#include <utility>

#include <kvikio/function_wrapper.hpp>

/**
 * @brief A simple, header-only thread pool that executes tasks in an embarrassingly parallel
 * manner.
 */
namespace kvikio {
class this_thread {
 public:
  static bool is_from_pool() { return get_thread_idx().has_value(); }

  static std::optional<std::size_t> get_thread_idx() { return this_thread_idx; }

 private:
  friend class ThreadPoolSimple;

  static void set_thread_idx(std::size_t thread_idx) { this_thread_idx = thread_idx; }

  inline static thread_local std::optional<std::size_t> this_thread_idx{std::nullopt};
};

struct Worker {
  std::thread thread;
  std::condition_variable task_available_cv;
  std::condition_variable task_done_cv;
  std::mutex task_mutex;
  std::queue<FunctionWrapper> task_queue;
  bool should_stop{false};
};

class ThreadPoolSimple {
 public:
  template <typename F>
  ThreadPoolSimple(unsigned int num_threads, F&& worker_thread_init_func)
    : _num_threads{num_threads}, _worker_thread_init_func{std::forward<F>(worker_thread_init_func)}
  {
    create_threads();
  }

  ThreadPoolSimple(unsigned int num_threads) : ThreadPoolSimple(num_threads, FunctionWrapper{}) {}

  ~ThreadPoolSimple() { destroy_threads(); }

  template <typename F>
  void reset(unsigned int num_threads, F&& worker_thread_init_func)
  {
    wait();
    destroy_threads();

    _num_threads             = num_threads;
    _worker_thread_init_func = std::forward<F>(worker_thread_init_func);
    create_threads();
  }

  void reset(unsigned int num_threads) { reset(num_threads, FunctionWrapper{}); }

  void wait()
  {
    for (unsigned int thread_idx = 0; thread_idx < _num_threads; ++thread_idx) {
      auto& task_done_cv = _workers[thread_idx].task_done_cv;
      auto& mut          = _workers[thread_idx].task_mutex;
      auto& task_queue   = _workers[thread_idx].task_queue;

      std::unique_lock lock(mut);
      task_done_cv.wait(lock, [&] { return task_queue.empty(); });
    }
  }

  unsigned int num_thread() const { return _num_threads; }

  template <typename F, typename R = std::invoke_result_t<std::decay_t<F>>>
  [[nodiscard]] std::future<R> submit_task(F&& task)
  {
    auto tid =
      std::atomic_fetch_add_explicit(&_task_submission_counter, 1, std::memory_order_relaxed);
    tid %= _num_threads;

    return submit_task_to_thread(std::forward<F>(task), tid);
  }

  template <typename F, typename R = std::invoke_result_t<std::decay_t<F>>>
  [[nodiscard]] std::future<R> submit_task_to_thread(F&& task, std::size_t thread_idx)
  {
    auto& task_available_cv = _workers[thread_idx].task_available_cv;
    auto& mut               = _workers[thread_idx].task_mutex;
    auto& task_queue        = _workers[thread_idx].task_queue;

    std::promise<R> p;
    auto fut = p.get_future();

    {
      std::lock_guard lock(mut);

      task_queue.emplace([task = std::forward<F>(task), p = std::move(p), thread_idx]() mutable {
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
    }

    task_available_cv.notify_one();
    return fut;
  }

 private:
  void run_worker(std::size_t thread_idx)
  {
    this_thread::set_thread_idx(thread_idx);

    auto& task_available_cv = _workers[thread_idx].task_available_cv;
    auto& task_done_cv      = _workers[thread_idx].task_done_cv;
    auto& mut               = _workers[thread_idx].task_mutex;
    auto& task_queue        = _workers[thread_idx].task_queue;
    auto& should_stop       = _workers[thread_idx].should_stop;

    if (_worker_thread_init_func) { std::invoke(_worker_thread_init_func); }

    while (true) {
      std::unique_lock lock(mut);

      if (task_queue.empty()) { task_done_cv.notify_all(); }

      task_available_cv.wait(lock, [&] { return !task_queue.empty() || should_stop; });

      if (should_stop) { break; }

      auto task = std::move(task_queue.front());
      task_queue.pop();
      lock.unlock();

      task();
    }
  }

  void create_threads()
  {
    _workers = std::make_unique<Worker[]>(_num_threads);
    for (unsigned int thread_idx = 0; thread_idx < _num_threads; ++thread_idx) {
      _workers[thread_idx].thread = std::thread([this, thread_idx] { run_worker(thread_idx); });
    }
  }

  void destroy_threads()
  {
    for (unsigned int thread_idx = 0; thread_idx < _num_threads; ++thread_idx) {
      auto& task_available_cv = _workers[thread_idx].task_available_cv;
      auto& mut               = _workers[thread_idx].task_mutex;

      {
        std::lock_guard lock(mut);
        _workers[thread_idx].should_stop = true;
      }

      task_available_cv.notify_one();

      _workers[thread_idx].thread.join();
    }
  }

  unsigned int _num_threads{};
  FunctionWrapper _worker_thread_init_func;
  std::unique_ptr<Worker[]> _workers;
  std::atomic_size_t _task_submission_counter{0};
};

}  // namespace kvikio
