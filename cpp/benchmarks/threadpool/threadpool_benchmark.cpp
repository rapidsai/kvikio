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
#include <vector>

#include <benchmark/benchmark.h>
#include <kvikio/defaults.hpp>
#include "kvikio/compat_mode.hpp"
#include "kvikio/file_handle.hpp"

void task_compute(std::size_t num_compute_iterations)
{
  [[maybe_unused]] double res{0.0};
  for (std::size_t i = 0u; i < num_compute_iterations; ++i) {
    auto x{static_cast<double>(i)};
    res += std::sqrt(x) + std::cbrt(x) + std::sin(x);
  }
}

void BM_threadpool_compute(benchmark::State& state)
{
  std::size_t const num_compute_iterations{100'000};
  std::size_t const num_compute_tasks{10'000};
  for (auto _ : state) {
    state.PauseTiming();
    kvikio::defaults::set_thread_pool_nthreads(state.range(0));

    state.ResumeTiming();
    for (std::size_t i = 0u; i < num_compute_tasks; ++i) {
      [[maybe_unused]] auto fut = kvikio::defaults::thread_pool().submit_task(
        [num_compute_iterations = num_compute_iterations] {
          task_compute(num_compute_iterations);
        });
    }
    kvikio::defaults::thread_pool().wait();
  }
}

void task_io() {}

void BM_threadpool_io(benchmark::State& state)
{
  kvikio::defaults::set_gds_threshold(0);
  kvikio::defaults::set_compat_mode(kvikio::CompatMode::ON);

  std::size_t const num_bytes{128ull * 1024ull * 1024ull};
  std::string file_path{"./test.bin"};
  std::vector<std::byte> buf(num_bytes, std::byte{0});
  kvikio::FileHandle fh{file_path, "w"};
  auto fut = fh.pwrite(buf.data(), num_bytes);
  fut.wait();

  for (auto _ : state) {
    state.PauseTiming();
    kvikio::defaults::set_thread_pool_nthreads(state.range(0));

    state.ResumeTiming();
    kvikio::FileHandle fh{file_path, "r"};
    auto fut = fh.pread(buf.data(), num_bytes);
    fut.wait();
  }
}

int main(int argc, char** argv)
{
  benchmark::Initialize(&argc, argv);

  benchmark::RegisterBenchmark("BM_threadpool_compute", BM_threadpool_compute)
    ->RangeMultiplier(2)
    ->Range(1, 64)
    ->Unit(benchmark::kMillisecond);

  benchmark::RegisterBenchmark("BM_threadpool_io", BM_threadpool_io)
    ->RangeMultiplier(2)
    ->Range(1, 64)
    ->Unit(benchmark::kMillisecond);

  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
}
