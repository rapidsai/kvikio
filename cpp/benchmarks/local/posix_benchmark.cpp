/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "posix_benchmark.hpp"
#include "../utils.hpp"

#include <fcntl.h>

#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>

#include <kvikio/compat_mode.hpp>
#include <kvikio/defaults.hpp>
#include <kvikio/detail/utils.hpp>
#include <kvikio/error.hpp>
#include <kvikio/file_handle.hpp>
#include <kvikio/file_utils.hpp>

namespace kvikio::benchmark {
void PosixConfig::parse_args(int argc, char** argv)
{
  Config::parse_args(argc, argv);
  static option long_options[] = {
    {"overwrite", no_argument, nullptr, 'w'}, {0, 0, 0, 0}
    // Sentinel to mark the end of the array. Needed by getopt_long()
  };

  int opt{0};
  int option_index{-1};

  // "f:"" means "-f" takes an argument
  // "c" means "-c" does not take an argument
  while ((opt = getopt_long(argc, argv, ":wp", long_options, &option_index)) != -1) {
    switch (opt) {
      case 'w': {
        overwrite_file = true;
        break;
      }
      case 'p': {
        per_drive_pool = true;
        break;
      }
      case ':': {
        // The parsed option has missing argument
        std::stringstream ss;
        ss << "Missing argument for option " << argv[optind - 1] << " (-"
           << static_cast<char>(optopt) << ")";
        throw std::runtime_error(ss.str());
        break;
      }
      default: {
        // Unknown option is deferred to subsequent parsing, if any
        break;
      }
    }
  }

  // Reset getopt state for second pass in the future
  optind = 1;
}

void PosixConfig::print_usage(std::string const& program_name)
{
  Config::print_usage(program_name);
  std::cout << "  -w, --overwrite         Overwrite existing file\n";
}

PosixBenchmark::PosixBenchmark(PosixConfig config) : Benchmark(std::move(config))
{
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

    // Initialize KvikIO setting
    if (_config.per_drive_pool) { kvikio::defaults::set_thread_pool_per_block_device(true); }

    // Initialize file
    // Create the file if the overwrite flag is on, or if the file does not exist.
    if (_config.overwrite_file || access(filepath.c_str(), F_OK) != 0) {
      kvikio::FileHandle file_handle(filepath, "w", kvikio::FileHandle::m644);
      auto fut = file_handle.pwrite(buf, _config.num_bytes);
      fut.get();
    }
  }
}

PosixBenchmark::~PosixBenchmark()
{
  for (auto&& buf : _bufs) {
    std::free(buf);
  }
}

void PosixBenchmark::initialize_impl()
{
  _file_handles.clear();

  for (auto const& filepath : _config.filepaths) {
    auto p = std::make_unique<kvikio::FileHandle>(filepath, "r");

    if (_config.o_direct) {
      auto file_status_flags = fcntl(p->fd(), F_GETFL);
      SYSCALL_CHECK(file_status_flags);
      SYSCALL_CHECK(fcntl(p->fd(), F_SETFL, file_status_flags | O_DIRECT));
    }

    _file_handles.push_back(std::move(p));
  }
}

void PosixBenchmark::cleanup_impl()
{
  for (auto&& file_handle : _file_handles) {
    file_handle->close();
  }
}

void PosixBenchmark::run_target_impl()
{
  std::vector<std::future<std::size_t>> futs;

  for (std::size_t i = 0; i < _file_handles.size(); ++i) {
    auto& file_handle = _file_handles[i];
    auto* buf         = _bufs[i];

    std::future<std::size_t> fut;
    if (_config.per_drive_pool) {
      fut = file_handle->pread(buf, _config.num_bytes);
    } else {
      fut = file_handle->pread(buf, _config.num_bytes);
    }
    futs.push_back(std::move(fut));
  }

  for (auto&& fut : futs) {
    fut.get();
  }
}

std::size_t PosixBenchmark::nbytes_impl() { return _config.num_bytes * _config.filepaths.size(); }
}  // namespace kvikio::benchmark

int main(int argc, char* argv[])
{
  try {
    kvikio::benchmark::PosixConfig config;
    config.parse_args(argc, argv);
    kvikio::benchmark::PosixBenchmark bench(std::move(config));
    bench.run();
  } catch (std::exception const& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }
  return 0;
}
