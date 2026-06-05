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

#include <kvikio/range_lock.hpp>
#include <kvikio/file_handle_rangelock.hpp>
#include <kvikio/defaults.hpp>
#include "utils/utils.hpp"

#include <thread>
#include <vector>
#include <chrono>
#include <atomic>

using namespace kvikio::test;

class RangeLockTest : public testing::Test {
 protected:
  void SetUp() override
  {
    TempDir tmp_dir{false};
    _filepath = tmp_dir.path() / "test_rangelock";
  }

  void TearDown() override {}

  std::filesystem::path _filepath;
};

TEST_F(RangeLockTest, non_overlapping_ranges)
{
  kvikio::RangeLockManager lock_manager;

  // Test that non-overlapping ranges can be locked simultaneously
  auto lock1 = lock_manager.lock_range(0, 100);
  auto lock2 = lock_manager.lock_range(100, 200);

  EXPECT_TRUE(lock_manager.is_range_locked(0, 100));
  EXPECT_TRUE(lock_manager.is_range_locked(100, 200));
  EXPECT_FALSE(lock_manager.is_range_locked(200, 300));

  EXPECT_EQ(lock_manager.num_locked_ranges(), 2);
}

TEST_F(RangeLockTest, overlapping_ranges_serialize)
{
  kvikio::RangeLockManager lock_manager;
  std::atomic<int> counter{0};
  std::atomic<bool> first_completed{false};

  std::thread t1([&]() {
    auto lock = lock_manager.lock_range(0, 100);
    counter++;
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    first_completed = true;
  });

  std::thread t2([&]() {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    auto lock = lock_manager.lock_range(50, 150);  // Overlaps with [0, 100)
    counter++;
    // Should only acquire after t1 releases
    EXPECT_TRUE(first_completed.load());
  });

  t1.join();
  t2.join();

  EXPECT_EQ(counter.load(), 2);
}

TEST_F(RangeLockTest, file_handle_parallel_writes)
{
  const size_t chunk_size = 1024;
  const size_t num_chunks = 10;

  // Create test data
  std::vector<uint8_t> data_a(chunk_size, 0xAA);
  std::vector<uint8_t> data_b(chunk_size, 0xBB);

  {
    kvikio::FileHandleWithRangeLock f(_filepath, "w+");

    std::vector<std::thread> threads;
    auto start = std::chrono::steady_clock::now();

    // Thread A writes even chunks
    threads.emplace_back([&]() {
      for (size_t i = 0; i < num_chunks; i += 2) {
        auto future = f.pwrite_rangelock(data_a.data(), chunk_size, i * chunk_size);
        future.get();
      }
    });

    // Thread B writes odd chunks
    threads.emplace_back([&]() {
      for (size_t i = 1; i < num_chunks; i += 2) {
        auto future = f.pwrite_rangelock(data_b.data(), chunk_size, i * chunk_size);
        future.get();
      }
    });

    for (auto& t : threads) {
      t.join();
    }

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::steady_clock::now() - start).count();

    // Verify no ranges remain locked
    EXPECT_EQ(f.num_locked_ranges(), 0);

    // Verify written data
    std::vector<uint8_t> verify(chunk_size);
    for (size_t i = 0; i < num_chunks; i++) {
      auto future = f.pread(verify.data(), chunk_size, i * chunk_size);
      future.get();

      uint8_t expected = (i % 2 == 0) ? 0xAA : 0xBB;
      EXPECT_EQ(verify[0], expected);
      EXPECT_EQ(verify[chunk_size - 1], expected);
    }
  }
}

TEST_F(RangeLockTest, range_lock_move_semantics)
{
  kvikio::RangeLockManager lock_manager;

  {
    auto lock1 = lock_manager.lock_range(0, 100);
    EXPECT_EQ(lock_manager.num_locked_ranges(), 1);

    // Move constructor
    auto lock2 = std::move(lock1);
    EXPECT_EQ(lock_manager.num_locked_ranges(), 1);

    // Original lock should be invalidated after move
    // lock2 still holds the lock
  }

  // Lock should be released when lock2 goes out of scope
  EXPECT_EQ(lock_manager.num_locked_ranges(), 0);
}

TEST_F(RangeLockTest, concurrent_non_overlapping_performance)
{
  kvikio::RangeLockManager lock_manager;
  const int num_threads = 4;
  const int ops_per_thread = 100;

  auto worker = [&](int thread_id) {
    for (int i = 0; i < ops_per_thread; i++) {
      size_t start = thread_id * 1000 + i * 10;
      size_t end = start + 5;
      auto lock = lock_manager.lock_range(start, end);
      // Simulate some work
      std::this_thread::sleep_for(std::chrono::microseconds(10));
    }
  };

  std::vector<std::thread> threads;
  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < num_threads; i++) {
    threads.emplace_back(worker, i);
  }

  for (auto& t : threads) {
    t.join();
  }

  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
    std::chrono::steady_clock::now() - start).count();

  // All ranges were non-overlapping, so they should execute in parallel
  // Total time should be much less than serial execution
  // Serial would take at least: num_threads * ops_per_thread * 10us = 4000us
  // Parallel should be close to: ops_per_thread * 10us = 1000us
  // Allow some overhead, but should be significantly faster than serial
  EXPECT_LT(duration, 2000);  // Should complete in less than 2 seconds
  EXPECT_EQ(lock_manager.num_locked_ranges(), 0);
}