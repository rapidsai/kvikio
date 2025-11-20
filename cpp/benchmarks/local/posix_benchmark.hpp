/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "../utils.hpp"

#include <getopt.h>
#include <memory>
#include <string>
#include <vector>

#include <kvikio/file_handle.hpp>

namespace kvikio::benchmark {

struct PosixConfig : Config {
  bool overwrite_file{false};

  virtual void parse_args(int argc, char** argv) override;
  virtual void print_usage(std::string const& program_name) override;
};

class PosixBenchmark : public Benchmark<PosixBenchmark, PosixConfig> {
  friend class Benchmark<PosixBenchmark, PosixConfig>;

 protected:
  std::vector<std::unique_ptr<kvikio::FileHandle>> _file_handles;
  std::vector<void*> _bufs;

  void initialize_impl();
  void cleanup_impl();
  void run_target_impl();

 public:
  PosixBenchmark(PosixConfig config);
  ~PosixBenchmark();
};
}  // namespace kvikio::benchmark
