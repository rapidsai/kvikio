/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

// This benchmark assesses the scalability of the thread pool.
//
// In the "strong scaling" study, the total amount of tasks is fixed, and the time to complete
// these tasks is evaluated as a function of thread count.
//
// In the "weak scaling" study, the expected amount of tasks per thread is fixed, and the total
// amount of tasks is then proportional to the thread count. Again, the time is evaluated as a
// function of thread count.

#include <cmath>
#include <cstdint>
#include <thread>
#include <vector>

#include <benchmark/benchmark.h>
#include <kvikio/defaults.hpp>
#include <kvikio/threadpool_simple.hpp>
#include <utils/utils.hpp>

namespace kvikio {
enum class ScalingType : uint8_t {
  STRONG_SCALING,
  WEAK_SCALING,
};

namespace constant {
std::size_t constexpr ntasks_strong_scaling{10'000};
std::size_t constexpr ntasks_weak_scaling{1'000};
std::size_t constexpr num_compute_iterations{1'000};
}  // namespace constant

void task_compute(std::size_t num_compute_iterations)
{
  [[maybe_unused]] double res{0.0};
  for (std::size_t i = 0u; i < num_compute_iterations; ++i) {
    auto x{static_cast<double>(i)};
    benchmark::DoNotOptimize(res += std::sqrt(x) + std::cbrt(x) + std::sin(x));
  }
}

template <ScalingType scaling_type>
void BM_BS_threadpool_compute(benchmark::State& state)
{
  auto const num_threads = state.range(0);

  std::size_t const num_compute_tasks = (scaling_type == ScalingType::STRONG_SCALING)
                                          ? constant::ntasks_strong_scaling
                                          : (constant::ntasks_weak_scaling * num_threads);

  kvikio::defaults::set_thread_pool_nthreads(num_threads);

  for (auto _ : state) {
    // Submit a total of "num_compute_tasks" tasks to the thread pool.
    for (auto i = std::size_t{0}; i < num_compute_tasks; ++i) {
      [[maybe_unused]] auto fut = kvikio::defaults::thread_pool().submit_task(
        [] { task_compute(constant::num_compute_iterations); });
    }
    kvikio::defaults::thread_pool().wait();
  }

  state.counters["threads"] = num_threads;
}

template <ScalingType scaling_type>
void BM_simple_threadpool_compute(benchmark::State& state)
{
  auto const num_threads = state.range(0);

  std::size_t const num_compute_tasks = (scaling_type == ScalingType::STRONG_SCALING)
                                          ? constant::ntasks_strong_scaling
                                          : (constant::ntasks_weak_scaling * num_threads);

  kvikio::ThreadPoolSimple thread_pool(num_threads);

  for (auto _ : state) {
    // Submit a total of "num_compute_tasks" tasks to the thread pool.
    for (auto i = std::size_t{0}; i < num_compute_tasks; ++i) {
      [[maybe_unused]] auto fut =
        thread_pool.submit_task([] { task_compute(constant::num_compute_iterations); });
    }
    thread_pool.wait();
  }

  state.counters["threads"] = num_threads;
}

template <ScalingType scaling_type>
void BM_static_task_compute(benchmark::State& state)
{
  auto const num_threads = state.range(0);

  for (auto _ : state) {
    std::vector<std::thread> threads(num_threads);
    for (auto&& thread : threads) {
      thread = std::thread([=] {
        std::size_t num_tasks_this_thread{};
        if constexpr (scaling_type == ScalingType::STRONG_SCALING) {
          auto const p = constant::ntasks_strong_scaling / num_threads;
          auto const q = constant::ntasks_strong_scaling % num_threads;
          num_tasks_this_thread =
            (static_cast<decltype(q)>(state.thread_index()) < q) ? (p + 1) : p;
        } else {
          num_tasks_this_thread = constant::ntasks_weak_scaling;
        }

        for (std::size_t i = 0; i < num_tasks_this_thread; ++i) {
          task_compute(constant::num_compute_iterations);
        }
      });
    }

    for (auto&& thread : threads) {
      thread.join();
    }
  }

  state.counters["threads"] = num_threads;
}
}  // namespace kvikio

int main(int argc, char** argv)
{
  benchmark::Initialize(&argc, argv);

  benchmark::RegisterBenchmark(
    "BS_threadpool_compute:strong_scaling",
    kvikio::BM_BS_threadpool_compute<kvikio::ScalingType::STRONG_SCALING>)
    ->RangeMultiplier(2)
    ->Range(1, 64)   // Increase from 1 to 64 (inclusive of both endpoints) with x2 stepping.
    ->UseRealTime()  // Use the wall clock to determine the number of benchmark iterations.
    ->Unit(benchmark::kMillisecond)
    ->MinTime(2);  // Minimum of 2 seconds.

  benchmark::RegisterBenchmark("BS_threadpool_compute:weak_scaling",
                               kvikio::BM_BS_threadpool_compute<kvikio::ScalingType::WEAK_SCALING>)
    ->RangeMultiplier(2)
    ->Range(1, 64)
    ->UseRealTime()
    ->Unit(benchmark::kMillisecond)
    ->MinTime(2);

  benchmark::RegisterBenchmark(
    "simple_threadpool_compute:strong_scaling",
    kvikio::BM_simple_threadpool_compute<kvikio::ScalingType::STRONG_SCALING>)
    ->RangeMultiplier(2)
    ->Range(1, 64)
    ->UseRealTime()
    ->Unit(benchmark::kMillisecond)
    ->MinTime(2);

  benchmark::RegisterBenchmark(
    "simple_threadpool_compute:weak_scaling",
    kvikio::BM_simple_threadpool_compute<kvikio::ScalingType::WEAK_SCALING>)
    ->RangeMultiplier(2)
    ->Range(1, 64)
    ->UseRealTime()
    ->Unit(benchmark::kMillisecond)
    ->MinTime(2);

  benchmark::RegisterBenchmark("static_task_compute:strong_scaling",
                               kvikio::BM_static_task_compute<kvikio::ScalingType::STRONG_SCALING>)
    ->RangeMultiplier(2)
    ->Range(1, 64)
    ->UseRealTime()
    ->Unit(benchmark::kMillisecond)
    ->MinTime(2);

  benchmark::RegisterBenchmark("static_task_compute:weak_scaling",
                               kvikio::BM_static_task_compute<kvikio::ScalingType::WEAK_SCALING>)
    ->RangeMultiplier(2)
    ->Range(1, 64)
    ->UseRealTime()
    ->Unit(benchmark::kMillisecond)
    ->MinTime(2);

  kvikio::utils::explain_default_metrics();

  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
}
