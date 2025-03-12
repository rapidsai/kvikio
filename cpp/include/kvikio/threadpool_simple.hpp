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
#include <functional>
#include <future>
#include <thread>
#include <type_traits>
#include <vector>

namespace kvikio {
/**
 * @brief A simple thread pool that executes tasks in an embarrassingly parallel manner.
 *
 * The implementation is header-only.
 */
class ThreadPoolSimple {
 public:
  ThreadPoolSimple(
    unsigned int num_threads, const std::function<void()>& worker_thread_init_func = [] {})
    : _num_threads{num_threads}, _worker_thread_init_func{worker_thread_init_func}
  {
  }

  void reset();

  template <typename F, typename R = std::invoke_result_t<std::decay_t<F>>>
  [[nodiscard]] std::future<R> submit_task(F&& task)
  {
  }

 private:
  void worker() {}

  void create_threads()
  {
    for (unsigned int i = 0; i < _num_threads; ++i) {
      _thread_container.emplace_back(&ThreadPoolSimple::worker, _worker_thread_init_func);
    }
  }

  void destroy_threads() {}

  std::atomic_bool _done{false};
  unsigned int _num_threads{};
  std::function<void()> _worker_thread_init_func{};
  std::vector<std::thread> _thread_container{};
};

}  // namespace kvikio