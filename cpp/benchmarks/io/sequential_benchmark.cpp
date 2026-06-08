/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "sequential_benchmark.hpp"
#include "common.hpp"
#include "kvikio/shim/cufile.hpp"

#include <fcntl.h>

#include <cuda_runtime.h>
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

void KvikIOSequentialConfig::parse_args(int argc, char** argv)
{
  Config::parse_args(argc, argv);
  static option long_options[] = {
    {0, 0, 0, 0}  // Sentinel to mark the end of the array. Needed by getopt_long()

  };

  int opt{0};
  int option_index{-1};

  while ((opt = getopt_long(argc, argv, "-:", long_options, &option_index)) != -1) {
    switch (opt) {
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
  optind = 0;
}

void KvikIOSequentialConfig::print_usage(std::string const& program_name)
{
  Config::print_usage(program_name);
}

KvikIOSequentialBenchmark::KvikIOSequentialBenchmark(KvikIOSequentialConfig const& config)
  : Benchmark(config)
{
  for (auto const& filepath : _config.filepaths) {
    _bufs.emplace_back(std::make_unique<Buffer<KvikIOSequentialConfig>>(_config));
    create_file(_config, filepath, *_bufs.back());
  }
}

KvikIOSequentialBenchmark::~KvikIOSequentialBenchmark() {}

void KvikIOSequentialBenchmark::initialize_impl()
{
  _file_handles.clear();

  for (auto const& filepath : _config.filepaths) {
    auto p = std::make_unique<FileHandle>(filepath, "r");

    if (_config.o_direct) {
      auto file_status_flags = fcntl(p->fd(), F_GETFL);
      SYSCALL_CHECK(file_status_flags);
      SYSCALL_CHECK(fcntl(p->fd(), F_SETFL, file_status_flags | O_DIRECT));
    }

    _file_handles.push_back(std::move(p));
  }
}

void KvikIOSequentialBenchmark::cleanup_impl()
{
  for (auto&& file_handle : _file_handles) {
    file_handle->close();
  }
}

void KvikIOSequentialBenchmark::run_target_impl()
{
  std::vector<std::future<std::size_t>> futs;

  for (std::size_t i = 0; i < _file_handles.size(); ++i) {
    auto fut = _file_handles[i]->pread(_bufs[i]->data(), _config.num_bytes);
    futs.push_back(std::move(fut));
  }

  for (auto&& fut : futs) {
    fut.get();
  }
}

std::size_t KvikIOSequentialBenchmark::nbytes_impl()
{
  return _config.num_bytes * _config.filepaths.size();
}

void CuFileSequentialConfig::parse_args(int argc, char** argv)
{
  Config::parse_args(argc, argv);
  static option long_options[] = {
    {0, 0, 0, 0}  // Sentinel to mark the end of the array. Needed by getopt_long()

  };

  int opt{0};
  int option_index{-1};

  while ((opt = getopt_long(argc, argv, "-:", long_options, &option_index)) != -1) {
    switch (opt) {
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
  optind = 0;
}

void CuFileSequentialConfig::print_usage(std::string const& program_name)
{
  Config::print_usage(program_name);
}

CuFileSequentialBenchmark::CuFileSequentialBenchmark(CuFileSequentialConfig const& config)
  : Benchmark(config)
{
  for (auto const& filepath : _config.filepaths) {
    _bufs.emplace_back(std::make_unique<Buffer<CuFileSequentialConfig>>(_config));
    create_file(_config, filepath, *_bufs.back());
  }
}

CuFileSequentialBenchmark::~CuFileSequentialBenchmark() {}

void CuFileSequentialBenchmark::initialize_impl()
{
  _file_handles.clear();

  for (auto const& filepath : _config.filepaths) {
    auto o_direct = _config.o_direct;
    auto p        = std::make_unique<CuFileHandle>(filepath, "r", o_direct, FileHandle::m644);
    _file_handles.push_back(std::move(p));
  }
}

void CuFileSequentialBenchmark::cleanup_impl()
{
  for (auto&& file_handle : _file_handles) {
    file_handle->close();
  }
}

void CuFileSequentialBenchmark::run_target_impl()
{
  for (std::size_t i = 0; i < _file_handles.size(); ++i) {
    off_t file_offset{0};
    off_t dev_ptr_offset{0};
    cuFileAPI::instance().Read(
      _file_handles[i]->handle(), _bufs[i]->data(), _config.num_bytes, file_offset, dev_ptr_offset);
  }
}

std::size_t CuFileSequentialBenchmark::nbytes_impl()
{
  return _config.num_bytes * _config.filepaths.size();
}
}  // namespace kvikio::benchmark

int main(int argc, char* argv[])
{
  try {
    auto backend = kvikio::benchmark::parse_backend(argc, argv);

    if (backend == kvikio::benchmark::Backend::FILEHANDLE) {
      kvikio::benchmark::KvikIOSequentialConfig config;
      config.parse_args(argc, argv);
      kvikio::benchmark::KvikIOSequentialBenchmark bench(config);
      bench.run();
    } else {
      kvikio::benchmark::CuFileSequentialConfig config;
      config.parse_args(argc, argv);
      kvikio::benchmark::CuFileSequentialBenchmark bench(config);
      bench.run();
    }
  } catch (std::exception const& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }
  return 0;
}
