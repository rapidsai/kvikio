/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuda_runtime.h>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include <kvikio/bounce_buffer.hpp>
#include <kvikio/defaults.hpp>
#include <kvikio/file_handle.hpp>
#include <kvikio/file_utils.hpp>

#define KVIKIO_CHECK_CUDA(err_code) kvikio::benchmark::check_cuda(err_code, __FILE__, __LINE__)

namespace kvikio::benchmark {
inline void check_cuda(cudaError_t err_code, const char* file, int line)
{
  if (err_code == cudaError_t::cudaSuccess) { return; }
  std::stringstream ss;
  int current_device{};
  cudaGetDevice(&current_device);
  ss << "CUDA runtime error on device " << current_device << ": " << cudaGetErrorName(err_code)
     << " (" << err_code << "): " << cudaGetErrorString(err_code) << " at " << file << ":" << line
     << "\n";
  throw std::runtime_error(ss.str());
}

enum Backend {
  FILEHANDLE,
  CUFILE,
};

Backend parse_backend(std::string const& str);
Backend parse_backend(int argc, char** argv);

// Helper to parse size strings like "1GiB", "1Gi", "1G".
std::size_t parse_size(std::string const& str);

bool parse_flag(std::string const& str);

struct Config {
  std::vector<std::string> filepaths;
  std::size_t num_bytes{4ull * 1024ull * 1024ull * 1024ull};
  unsigned int num_threads{1};
  bool use_gpu_buffer{false};
  int gpu_index{0};
  int repetition{5};
  bool overwrite_file{false};
  bool o_direct{true};
  bool align_buffer{true};
  bool drop_file_cache{false};
  bool open_file_once{false};

  virtual void parse_args(int argc, char** argv);
  virtual void print_usage(std::string const& program_name);
};

template <typename Derived, typename ConfigType>
class Benchmark {
 protected:
  ConfigType const& _config;

  void initialize() { static_cast<Derived*>(this)->initialize_impl(); }
  void cleanup() { static_cast<Derived*>(this)->cleanup_impl(); }
  void run_target() { static_cast<Derived*>(this)->run_target_impl(); }
  std::size_t nbytes() { return static_cast<Derived*>(this)->nbytes_impl(); }

 public:
  Benchmark(ConfigType const& config) : _config(config)
  {
    defaults::set_thread_pool_nthreads(_config.num_threads);
  }

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
      }
      double bandwidth = nbytes() / time_elapsed_us * 1e6 / 1024.0 / 1024.0;
      std::cout << std::string(4, ' ') << std::left << std::setw(4) << idx << std::setw(10)
                << bandwidth << " [MiB/s]" << std::endl;

      if (!_config.open_file_once) { cleanup(); }
    }
    double average_bandwidth = nbytes() * count / time_elapsed_total_us * 1e6 / 1024.0 / 1024.0;
    std::cout << std::string(4, ' ') << "Average bandwidth: " << std::setw(10) << average_bandwidth
              << " [MiB/s]" << std::endl;

    if (_config.open_file_once) { cleanup(); }
  }
};

class CudaPageAlignedDeviceAllocator {
 public:
  void* allocate(std::size_t size);

  void deallocate(void* buffer, std::size_t size);
};

template <typename ConfigType>
class Buffer {
 public:
  Buffer(ConfigType config) : _config(config) { allocate(); }
  ~Buffer() { deallocate(); }

  Buffer(Buffer const&)            = delete;
  Buffer& operator=(Buffer const&) = delete;

  Buffer(Buffer&& o) noexcept
    : _config(std::exchange(o._config, {})),
      _data(std::exchange(o._data, {})),
      _original_data(std::exchange(o._original_data, {}))
  {
  }

  Buffer& operator=(Buffer&& o) noexcept
  {
    if (this == &o) { return *this; }
    deallocate();
    _config        = std::exchange(o._config, {});
    _data          = std::exchange(o._data, {});
    _original_data = std::exchange(o._original_data, {});
  }

  void* data() const { return _data; }
  void* size() const { return _size; }

 private:
  void allocate()
  {
    if (_config.use_gpu_buffer) {
      KVIKIO_CHECK_CUDA(cudaSetDevice(_config.gpu_index));
      if (_config.align_buffer) {
        CudaPageAlignedDeviceAllocator alloc;
        _original_data = alloc.allocate(_config.num_bytes);
      } else {
        KVIKIO_CHECK_CUDA(cudaMalloc(&_original_data, _config.num_bytes));
      }
    } else {
      if (_config.align_buffer) {
        PageAlignedAllocator alloc;
        _original_data = alloc.allocate(_config.num_bytes);
      } else {
        _original_data = std::malloc(_config.num_bytes);
      }
    }

    _data = _original_data;
  }

  void deallocate()
  {
    if (_config.use_gpu_buffer) {
      if (_config.align_buffer) {
      } else {
        KVIKIO_CHECK_CUDA(cudaFree(_original_data));
      }
    } else {
      std::free(_original_data);
    }

    _data          = nullptr;
    _original_data = nullptr;
    _size          = 0;
  }

  ConfigType _config;
  void* _data{};
  void* _original_data{};
  std::size_t _size{};
};

template <typename ConfigType>
void create_file(ConfigType const& config,
                 std::string const& filepath,
                 Buffer<ConfigType> const& buf)
{
  // Create the file if the overwrite flag is on, or if the file does not exist, or if the file size
  // is wrong.
  if (config.overwrite_file || access(filepath.c_str(), F_OK) != 0 ||
      get_file_size(filepath) != config.num_bytes) {
    FileHandle file_handle(filepath, "w", FileHandle::m644);
    auto fut = file_handle.pwrite(buf.data(), config.num_bytes);
    fut.get();
  }
}

class CuFileHandle {
 public:
  CuFileHandle(std::string const& file_path, std::string const& flags, bool o_direct, mode_t mode);
  ~CuFileHandle() = default;

  CuFileHandle(CuFileHandle const&)            = delete;
  CuFileHandle& operator=(CuFileHandle const&) = delete;

  CuFileHandle(CuFileHandle&&) noexcept   = default;
  CuFileHandle& operator=(CuFileHandle&&) = default;

  void close();

  CUfileHandle_t handle() const noexcept;

 private:
  FileWrapper _file_wrapper;
  CUFileHandleWrapper _cufile_handle_wrapper;
};
}  // namespace kvikio::benchmark
