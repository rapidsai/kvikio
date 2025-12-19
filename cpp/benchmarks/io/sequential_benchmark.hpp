/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "common.hpp"

#include <getopt.h>
#include <memory>
#include <string>
#include <vector>

#include <kvikio/file_handle.hpp>
#include <kvikio/threadpool_wrapper.hpp>

namespace kvikio::benchmark {

struct SequentialConfig : Config {
  virtual void parse_args(int argc, char** argv) override;
  virtual void print_usage(std::string const& program_name) override;
};

class SequentialBenchmark : public Benchmark<SequentialBenchmark, SequentialConfig> {
  friend class Benchmark<SequentialBenchmark, SequentialConfig>;

 protected:
  std::vector<std::unique_ptr<kvikio::FileHandle>> _file_handles;
  std::vector<void*> _bufs;
  std::vector<std::unique_ptr<kvikio::ThreadPool>> _thread_pools;

  void initialize_impl();
  void cleanup_impl();
  void run_target_impl();
  std::size_t nbytes_impl();

 public:
  SequentialBenchmark(SequentialConfig config);
  ~SequentialBenchmark();
};
}  // namespace kvikio::benchmark
