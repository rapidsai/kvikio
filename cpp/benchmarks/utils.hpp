/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <chrono>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include <kvikio/file_utils.hpp>

namespace kvikio::benchmark {
// Helper to parse size strings like "1GiB", "1Gi", "1G".
std::size_t parse_size(std::string const& str);

struct Config {
  std::size_t num_bytes{4ull * 1024ull * 1024ull * 1024ull};
  std::vector<std::string> filepaths;
  bool align_buffer{true};
  bool o_direct{true};
  bool drop_file_cache{false};
  bool compat_mode{true};
  unsigned int num_threads{1};
  bool open_file_once{false};
  int repetition{5};

  virtual void parse_args(int argc, char** argv);
  virtual void print_usage(std::string const& program_name);
};

template <typename Derived, typename ConfigType>
class Benchmark {
 protected:
  ConfigType _config;

  void initialize() { static_cast<Derived*>(this)->initialize_impl(); }
  void cleanup() { static_cast<Derived*>(this)->cleanup_impl(); }
  void run_target() { static_cast<Derived*>(this)->run_target_impl(); }

 public:
  Benchmark(ConfigType config) : _config(std::move(config)) {}

  void run()
  {
    if (_config.open_file_once) { initialize(); }

    decltype(_config.repetition) count{0};
    double time_elapsed_total_us{0.0};
    for (decltype(_config.repetition) idx = 0; idx < _config.repetition; ++idx) {
      if (_config.drop_file_cache) { kvikio::clear_page_cache(); }

      if (!_config.open_file_once) { initialize(); }

      auto start = std::chrono::steady_clock::now();
      run_target();
      auto end = std::chrono::steady_clock::now();

      std::chrono::duration<double, std::micro> time_elapsed = end - start;
      double time_elapsed_us                                 = time_elapsed.count();
      if (idx > 0) {
        ++count;
        time_elapsed_total_us += time_elapsed_us;
        double bandwidth = _config.num_bytes / time_elapsed_us * 1e6 / 1024.0 / 1024.0;
        std::cout << std::string(4, ' ') << std::left << std::setw(4) << idx << std::setw(10)
                  << bandwidth << " [MiB/s]" << std::endl;
      }

      if (!_config.open_file_once) { cleanup(); }
    }
    double average_bandwidth =
      _config.num_bytes * count / time_elapsed_total_us * 1e6 / 1024.0 / 1024.0;
    std::cout << std::string(4, ' ') << "Average bandwidth: " << std::setw(10) << average_bandwidth
              << " [MiB/s]" << std::endl;

    if (_config.open_file_once) { cleanup(); }
  }
};
}  // namespace kvikio::benchmark
