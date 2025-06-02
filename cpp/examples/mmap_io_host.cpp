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

#include <chrono>
#include <iostream>
#include <ratio>
#include <string>
#include <vector>

#include <kvikio/file_handle.hpp>
#include <kvikio/file_utils.hpp>
#include <kvikio/mmap.hpp>

std::string parse_cmd(int argc, char* argv[]) { return (argc > 1) ? argv[1] : "/tmp"; }

class Timer {
 public:
  void start() { _start = std::chrono::high_resolution_clock::now(); }
  double elapsed_time()
  {
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::micro> time_elapsed = end - _start;
    auto result                                            = time_elapsed.count();
    std::cout << "    Elapsed time: " << result << " us\n";
    return result;
  }

 private:
  std::chrono::time_point<std::chrono::high_resolution_clock> _start;
};

class IoHostManager {
 public:
  IoHostManager(std::string const& test_dir = "/tmp", bool clear_page_cache = false)
    : _test_filepath(test_dir + "/test-file"),
      _clear_page_cache(clear_page_cache),
      _data_size(1024ull * 1024ull * 1024ull),
      _num_repetition(10)
  {
    std::vector<std::byte> v(_data_size, {});
    kvikio::FileHandle file_handle(_test_filepath, "w");
    auto fut = file_handle.pwrite(v.data(), v.size());
    fut.get();
  }

  void use_standard_io_parallel()
  {
    std::cout << "Standard I/O\n";
    std::vector<std::byte> v(_data_size, {});
    double ave_init_time{0.0};
    double ave_io_time{0.0};

    for (std::size_t i = 0; i < _num_repetition; ++i) {
      if (_clear_page_cache) { kvikio::clear_page_cache(); }
      print_page_cache_info();

      Timer timer;
      timer.start();
      kvikio::FileHandle file_handle(_test_filepath, "r");
      ave_init_time += timer.elapsed_time();

      timer.start();
      auto fut = file_handle.pread(v.data(), _data_size);
      fut.get();
      ave_io_time += timer.elapsed_time();
    }

    std::cout << "    Average initialization time: " << ave_init_time / _num_repetition << "\n";
    std::cout << "    Average I/O time: " << ave_io_time / _num_repetition << "\n";
  }

  void use_mmap_io_seq()
  {
    std::cout << "Mmap I/O (sequential prefault)\n";
    double ave_init_time{0.0};
    double ave_io_time{0.0};

    for (std::size_t i = 0; i < _num_repetition; ++i) {
      if (_clear_page_cache) { kvikio::clear_page_cache(); }
      print_page_cache_info();

      Timer timer;
      timer.start();
      kvikio::MmapHandle mmap_handle(_test_filepath, "r");
      ave_init_time += timer.elapsed_time();

      timer.start();
      mmap_handle.read(_data_size, 0, true);
      ave_io_time += timer.elapsed_time();
    }

    std::cout << "    Average initialization time: " << ave_init_time / _num_repetition << "\n";
    std::cout << "    Average I/O time: " << ave_io_time / _num_repetition << "\n";
  }

  void use_mmap_io_parallel()
  {
    std::cout << "Mmap I/O (parallel prefault)\n";
    double ave_init_time{0.0};
    double ave_io_time{0.0};

    for (std::size_t i = 0; i < _num_repetition; ++i) {
      if (_clear_page_cache) { kvikio::clear_page_cache(); }
      print_page_cache_info();

      Timer timer;
      timer.start();
      kvikio::MmapHandle mmap_handle(_test_filepath, "r");
      ave_init_time += timer.elapsed_time();

      timer.start();
      auto res = mmap_handle.pread(_data_size, 0, true);
      res.second.get();
      ave_io_time += timer.elapsed_time();
    }

    std::cout << "    Average initialization time: " << ave_init_time / _num_repetition << "\n";
    std::cout << "    Average I/O time: " << ave_io_time / _num_repetition << "\n";
  }

 private:
  void print_page_cache_info()
  {
    auto const [num_pages_in_page_cache, num_pages] = kvikio::get_page_cache_info(_test_filepath);
    std::cout << "    Page cache residency ratio: "
              << static_cast<double>(num_pages_in_page_cache) / static_cast<double>(num_pages)
              << "\n";
  }
  std::string _test_filepath;
  bool _clear_page_cache;
  std::size_t _data_size;
  std::size_t _num_repetition;
};

int main()
{
  //   auto test_dir = parse_cmd(argc, argv);
  IoHostManager io_host_manager{"/mnt/nvme", true};

  io_host_manager.use_mmap_io_seq();

  io_host_manager.use_mmap_io_parallel();

  io_host_manager.use_standard_io_parallel();

  return 0;
}
