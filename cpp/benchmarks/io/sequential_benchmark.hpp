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
#include <kvikio/file_utils.hpp>

namespace kvikio::benchmark {

struct KvikIOSequentialConfig : Config {
  virtual void parse_args(int argc, char** argv) override;
  virtual void print_usage(std::string const& program_name) override;
};

class KvikIOSequentialBenchmark
  : public Benchmark<KvikIOSequentialBenchmark, KvikIOSequentialConfig> {
  friend class Benchmark<KvikIOSequentialBenchmark, KvikIOSequentialConfig>;

 protected:
  std::vector<std::unique_ptr<FileHandle>> _file_handles;
  std::vector<std::unique_ptr<Buffer<KvikIOSequentialConfig>>> _bufs;

  void initialize_impl();
  void cleanup_impl();
  void run_target_impl();
  std::size_t nbytes_impl();

 public:
  KvikIOSequentialBenchmark(KvikIOSequentialConfig const& config);
  ~KvikIOSequentialBenchmark();
};

struct CuFileSequentialConfig : Config {
  virtual void parse_args(int argc, char** argv) override;
  virtual void print_usage(std::string const& program_name) override;
};

class CuFileSequentialBenchmark
  : public Benchmark<CuFileSequentialBenchmark, CuFileSequentialConfig> {
  friend class Benchmark<CuFileSequentialBenchmark, CuFileSequentialConfig>;

 protected:
  std::vector<std::unique_ptr<CuFileHandle>> _file_handles;
  std::vector<std::unique_ptr<Buffer<CuFileSequentialConfig>>> _bufs;

  void initialize_impl();
  void cleanup_impl();
  void run_target_impl();
  std::size_t nbytes_impl();

 public:
  CuFileSequentialBenchmark(CuFileSequentialConfig const& config);
  ~CuFileSequentialBenchmark();
};

}  // namespace kvikio::benchmark
