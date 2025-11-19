/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <getopt.h>
#include <memory>
#include <string>
#include <vector>

#include <kvikio/file_handle.hpp>

namespace kvikio::benchmark {

struct Config {
  std::size_t num_bytes{4ull * 1024ull * 1024ull * 1024ull};
  std::vector<std::string> filepaths;
  bool overwrite_file{false};
  bool align_buffer{true};
  bool o_direct{true};
  bool drop_file_cache{false};
  bool compat_mode{true};
  unsigned int num_threads{1};
  bool open_file_once{false};
  int repetition{5};

  static Config parse_args(int argc, char** argv);
  static void print_usage(std::string const& program_name);
};

class BenchmarkManager {
 private:
  Config const& _config;
  std::vector<std::unique_ptr<kvikio::FileHandle>> _file_handles;
  std::vector<void*> _bufs;

  void initialize();
  void cleanup();
  void run_target();

 public:
  BenchmarkManager(Config const& config);
  ~BenchmarkManager();
  void run();
};
}  // namespace kvikio::benchmark
