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

#include <cmath>

#include <benchmark/benchmark.h>
#include <kvikio/defaults.hpp>

void compute()
{
  std::size_t const num_compute_iterations{100'000u};
  [[maybe_unused]] double res{0.0};
  for (std::size_t i = 0u; i < num_compute_iterations; ++i) {
    auto x{static_cast<double>(i)};
    res += std::sqrt(x) + std::cbrt(x) + std::sin(x);
  }
}

void BM_threadpool_compute(benchmark::State& state)
{
  std::size_t const num_compute_tasks{10'000u};
  for (auto _ : state) {
    state.PauseTiming();
    kvikio::defaults::set_thread_pool_nthreads(state.range(0));

    state.ResumeTiming();
    for (std::size_t i = 0u; i < num_compute_tasks; ++i) {
      [[maybe_unused]] auto fut = kvikio::defaults::thread_pool().submit_task(compute);
    }
    kvikio::defaults::thread_pool().wait();
  }
}

int main(int argc, char** argv)
{
  benchmark::Initialize(&argc, argv);
  benchmark::RegisterBenchmark("BM_threadpool_compute", BM_threadpool_compute)
    ->RangeMultiplier(2)
    ->Range(1, 64)
    ->Unit(benchmark::kMillisecond);
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
}
