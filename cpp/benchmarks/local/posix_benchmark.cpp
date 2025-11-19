/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "posix_benchmark.hpp"

#include <fcntl.h>
#include <unistd.h>
#include <algorithm>
#include <chrono>
#include <exception>
#include <iomanip>
#include <iostream>

#include <kvikio/compat_mode.hpp>
#include <kvikio/detail/utils.hpp>
#include <kvikio/file_handle.hpp>
#include <kvikio/file_utils.hpp>
#include <ratio>
#include "kvikio/error.hpp"

namespace kvikio::benchmark {

// Helper to parse size strings like "1G", "500M"
std::size_t parse_size(std::string const& str)
{
  if (str.empty()) { throw std::invalid_argument("Empty size string"); }

  // Parse the numeric part
  std::size_t pos{};
  double value{};
  try {
    value = std::stod(str, &pos);
  } catch (std::exception const& e) {
    throw std::invalid_argument("Invalid size format: " + str);
  }

  if (value < 0) { throw std::invalid_argument("Size cannot be negative"); }

  // Extract suffix (everything after the number)
  auto suffix = str.substr(pos);

  // No suffix means raw bytes
  if (suffix.empty()) { return static_cast<std::size_t>(value); }

  // Normalize to uppercase for case-insensitive comparison
  std::transform(
    suffix.begin(), suffix.end(), suffix.begin(), [](unsigned char c) { return std::tolower(c); });

  // All multipliers use 1024 (binary), not 1000
  std::size_t multiplier{1};

  // Support both K/Ki, M/Mi, etc. as synonyms (all 1024-based)
  if (suffix == "K" || suffix == "KI") {
    multiplier = 1024ULL;
  } else if (suffix == "M" || suffix == "MI") {
    multiplier = 1024ULL * 1024;
  } else if (suffix == "G" || suffix == "GI") {
    multiplier = 1024ULL * 1024 * 1024;
  } else if (suffix == "T" || suffix == "TI") {
    multiplier = 1024ULL * 1024 * 1024 * 1024;
  } else {
    throw std::invalid_argument("Invalid size suffix: '" + suffix +
                                "' (use K/Ki, M/Mi, G/Gi, or T/Ti)");
  }

  return static_cast<std::size_t>(value * multiplier);
}

Config Config::parse_args(int argc, char** argv)
{
  Config config;

  static struct option long_options[] = {{"file", required_argument, 0, 'f'},
                                         {"size", required_argument, 0, 's'},
                                         {"threads", required_argument, 0, 't'},
                                         {"repetitions", required_argument, 0, 'r'},
                                         {"no-direct", no_argument, 0, 'D'},
                                         {"no-align", no_argument, 0, 'A'},
                                         {"drop-cache", no_argument, 0, 'c'},
                                         {"overwrite", no_argument, 0, 'w'},
                                         {"open-once", no_argument, 0, 'o'},
                                         {"help", no_argument, 0, 'h'},
                                         {0, 0, 0, 0}};

  int opt;
  int option_index = 0;

  while ((opt = getopt_long(argc, argv, "f:s:t:r:DAcwoh", long_options, &option_index)) != -1) {
    switch (opt) {
      case 'f': config.filepaths.push_back(optarg); break;
      case 's':
        config.num_bytes = parse_size(optarg);  // Helper to parse "1G", "500M", etc.
        break;
      case 't': config.num_threads = std::stoul(optarg); break;
      case 'r': config.repetition = std::stoi(optarg); break;
      case 'D': config.o_direct = false; break;
      case 'A': config.align_buffer = false; break;
      case 'c': config.drop_file_cache = true; break;
      case 'w': config.overwrite_file = true; break;
      case 'o': config.open_file_once = true; break;
      case 'h': config.print_usage(argv[0]); std::exit(0);
      default: config.print_usage(argv[0]); std::exit(1);
    }
  }

  // Validation
  if (config.filepaths.empty()) { throw std::invalid_argument("--file is required"); }

  return config;
}

void Config::print_usage(std::string const& program_name)
{
  std::cout << "Usage: " << program_name << " [OPTIONS]\n\n"
            << "Options:\n"
            << "  -f, --file PATH         File path to benchmark (required)\n"
            << "  -s, --size SIZE         Number of bytes to read (default: 4G)\n"
            << "                          Supports suffixes: K, M, G, T\n"
            << "  -t, --threads NUM       Number of threads (default: 1)\n"
            << "  -r, --repetitions NUM   Number of repetitions (default: 5)\n"
            << "  -D, --no-direct         Disable O_DIRECT (use buffered I/O)\n"
            << "  -A, --no-align          Disable buffer alignment\n"
            << "  -c, --drop-cache        Drop page cache before each run\n"
            << "  -w, --overwrite         Overwrite existing file\n"
            << "  -o, --open-once         Open file once (not per iteration)\n"
            << "  -h, --help              Show this help message\n\n"
            << "Examples:\n"
            << "  " << program_name << " -f /mnt/nvme/test.bin -s 1G -t 4\n"
            << "  " << program_name
            << " --file /dev/nvme0n1 --size 10G --threads 16 --drop-cache\n";
}

BenchmarkManager::BenchmarkManager(Config const& config) : _config(config)
{
  kvikio::defaults::set_thread_pool_nthreads(_config.num_threads);

  for (auto const& filepath : _config.filepaths) {
    // Initialize buffer
    void* buf{};

    if (_config.align_buffer) {
      auto const page_size    = get_page_size();
      auto const aligned_size = kvikio::detail::align_up(_config.num_bytes, page_size);
      buf                     = std::aligned_alloc(page_size, aligned_size);
    } else {
      buf = std::malloc(_config.num_bytes);
    }

    std::memset(buf, 0, _config.num_bytes);

    _bufs.push_back(buf);

    // Initialize file
    // Create the file if the overwrite flag is on, or if the file does not exist.
    if (_config.overwrite_file || access(filepath.c_str(), F_OK) != 0) {
      kvikio::FileHandle file_handle(
        filepath, "w", kvikio::FileHandle::m644, kvikio::CompatMode::ON);
      auto fut = file_handle.pwrite(buf, _config.num_bytes);
      fut.get();
    }
  }
}

BenchmarkManager::~BenchmarkManager()
{
  for (auto&& buf : _bufs) {
    std::free(buf);
  }
}

void BenchmarkManager::run()
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

void BenchmarkManager::initialize()
{
  _file_handles.clear();

  for (auto const& filepath : _config.filepaths) {
    auto p = std::make_unique<kvikio::FileHandle>(
      filepath, "r", kvikio::FileHandle::m644, kvikio::CompatMode::ON);

    if (_config.o_direct) {
      auto file_status_flags = fcntl(p->fd(), F_GETFL);
      SYSCALL_CHECK(file_status_flags);
      SYSCALL_CHECK(fcntl(p->fd(), F_SETFL, file_status_flags | O_DIRECT));
    }

    _file_handles.push_back(std::move(p));
  }
}

void BenchmarkManager::cleanup()
{
  for (auto&& file_handle : _file_handles) {
    file_handle->close();
  }
}

void BenchmarkManager::run_target()
{
  std::vector<std::future<std::size_t>> futs;

  for (std::size_t i = 0; i < _file_handles.size(); ++i) {
    auto& file_handle = _file_handles[i];
    auto* buf         = _bufs[i];
    auto fut          = file_handle->pread(buf, _config.num_bytes);
    futs.push_back(std::move(fut));
  }

  for (auto&& fut : futs) {
    fut.get();
  }
}
}  // namespace kvikio::benchmark

int main(int argc, char* argv[])
{
  try {
    auto config = kvikio::benchmark::Config::parse_args(argc, argv);
    kvikio::benchmark::BenchmarkManager bench_manager(config);
    bench_manager.run();
  } catch (std::exception const& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }
  return 0;
}
