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
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// This benchmark assesses the performance of the thread pool.
// In the "strong scaling" study, the total amount of tasks is fixed, and the time to complete
// these tasks is evaluated as a function of thread count.
// In the "weak scaling" study, the expected tasks per thread is fixed, and the total amount of
// tasks is then proportional to the thread count. Again, the time is evaluated as a function of
// thread count.

#include <cmath>
#include <cstdint>

#include <benchmark/benchmark.h>
#include <kvikio/defaults.hpp>

namespace kvikio {
enum class ScalingType : uint8_t {
  StrongScaling,
  WeakScaling,
};

void task_compute(std::size_t num_compute_iterations)
{
  [[maybe_unused]] double res{0.0};
  for (std::size_t i = 0u; i < num_compute_iterations; ++i) {
    auto x{static_cast<double>(i)};
    res += std::sqrt(x) + std::cbrt(x) + std::sin(x);
  }
}

template <ScalingType scaling_type>
void BM_threadpool_compute(benchmark::State& state)
{
  auto num_threads = state.range(0);

  std::string label;
  std::size_t num_compute_tasks;
  if constexpr (scaling_type == ScalingType::StrongScaling) {
    num_compute_tasks = 10'000;
  } else {
    num_compute_tasks = 1'000 * num_threads;
  }

  std::size_t const num_compute_iterations{100'000};
  kvikio::defaults::set_thread_pool_nthreads(num_threads);

  for (auto _ : state) {
    for (std::size_t i = 0u; i < num_compute_tasks; ++i) {
      [[maybe_unused]] auto fut = kvikio::defaults::thread_pool().submit_task(
        [num_compute_iterations = num_compute_iterations] {
          task_compute(num_compute_iterations);
        });
    }
    kvikio::defaults::thread_pool().wait();
  }

  state.counters["threads"] = num_threads;
}
}  // namespace kvikio

int main(int argc, char** argv)
{
  benchmark::Initialize(&argc, argv);

  benchmark::RegisterBenchmark("BM_threadpool_compute:strong_scaling",
                               kvikio::BM_threadpool_compute<kvikio::ScalingType::StrongScaling>)
    ->RangeMultiplier(2)
    ->Range(1, 64)
    ->Unit(benchmark::kMillisecond);

  benchmark::RegisterBenchmark("BM_threadpool_compute:weak_scaling",
                               kvikio::BM_threadpool_compute<kvikio::ScalingType::WeakScaling>)
    ->RangeMultiplier(2)
    ->Range(1, 64)
    ->Unit(benchmark::kMillisecond);

  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
}
