/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "common.hpp"

#include <getopt.h>

#include <algorithm>
#include <sstream>
#include <stdexcept>
#include "kvikio/detail/utils.hpp"
#include "kvikio/utils.hpp"

namespace kvikio::benchmark {

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

  // Normalize to lowercase for case-insensitive comparison
  std::transform(
    suffix.begin(), suffix.end(), suffix.begin(), [](unsigned char c) { return std::tolower(c); });

  // All multipliers use 1024 (binary), not 1000
  std::size_t multiplier{1};

  // Support both K/Ki, M/Mi, etc. as synonyms (all 1024-based)
  std::size_t constexpr one_Ki{1024ULL};
  std::size_t constexpr one_Mi{1024ULL * one_Ki};
  std::size_t constexpr one_Gi{1024ULL * one_Mi};
  std::size_t constexpr one_Ti{1024ULL * one_Gi};
  std::size_t constexpr one_Pi{1024ULL * one_Ti};
  if (suffix == "k" || suffix == "ki" || suffix == "kb" || suffix == "kib") {
    multiplier = one_Ki;
  } else if (suffix == "m" || suffix == "mi" || suffix == "mb" || suffix == "mib") {
    multiplier = one_Mi;
  } else if (suffix == "g" || suffix == "gi" || suffix == "gb" || suffix == "gib") {
    multiplier = one_Gi;
  } else if (suffix == "t" || suffix == "ti" || suffix == "tb" || suffix == "tib") {
    multiplier = one_Ti;
  } else if (suffix == "p" || suffix == "pi" || suffix == "pb" || suffix == "pib") {
    multiplier = one_Pi;
  } else {
    throw std::invalid_argument(
      "Invalid size suffix: '" + suffix +
      "' (use K/Ki/KB/KiB, M/Mi/MB/MiB, G/Gi/GB/GiB, T/Ti/TB/TiB, or P/Pi/PB/PiB)");
  }

  return static_cast<std::size_t>(value * multiplier);
}

bool parse_flag(std::string const& str)
{
  if (str.empty()) { throw std::invalid_argument("Empty flag"); }

  // Normalize to lowercase for case-insensitive comparison
  auto result{str};
  std::transform(
    result.begin(), result.end(), result.begin(), [](unsigned char c) { return std::tolower(c); });

  if (result == "true" || result == "on" || result == "yes" || result == "1") {
    return true;
  } else if (result == "false" || result == "off" || result == "no" || result == "0") {
    return false;
  } else {
    throw std::invalid_argument("Invalid flag: '" + str +
                                "' (use true/false, on/off, yes/no, or 1/0)");
  }
}

void Config::parse_args(int argc, char** argv)
{
  enum LongOnlyOpts {
    O_DIRECT = 1000,
    ALIGN_BUFFER,
    DROP_CACHE,
    OPEN_ONCE,
  };

  static option long_options[] = {
    {"file", required_argument, nullptr, 'f'},
    {"size", required_argument, nullptr, 's'},
    {"threads", required_argument, nullptr, 't'},
    {"use-gpu-buffer", required_argument, nullptr, 'g'},
    {"gpu-index", required_argument, nullptr, 'd'},
    {"repetitions", required_argument, nullptr, 'r'},
    {"o-direct", required_argument, nullptr, LongOnlyOpts::O_DIRECT},
    {"align-buffer", required_argument, nullptr, LongOnlyOpts::ALIGN_BUFFER},
    {"drop-cache", required_argument, nullptr, LongOnlyOpts::DROP_CACHE},
    {"overwrite", required_argument, nullptr, 'w'},
    {"open-once", required_argument, nullptr, LongOnlyOpts::OPEN_ONCE},
    {"help", no_argument, nullptr, 'h'},
    {0, 0, 0, 0}  // Sentinel to mark the end of the array. Needed by getopt_long()
  };

  int opt{0};
  int option_index{-1};

  // - By default getopt_long() returns '?' to indicate errors if an option has missing argument or
  // if an unknown option is encountered. The starting ':' in the optstring modifies this behavior.
  // Missing argument error now causes the return value to be ':'. Unknow option still leads to '?'
  // and its processing is deferred.
  // - "f:" means option "-f" takes an argument
  // - "c" means option "-c" does not take an argument
  while ((opt = getopt_long(argc, argv, ":f:s:t:g:d:r:w:h", long_options, &option_index)) != -1) {
    switch (opt) {
      case 'f': {
        filepaths.push_back(optarg);
        break;
      }
      case 's': {
        num_bytes = parse_size(optarg);  // Helper to parse "1G", "500M", etc.
        break;
      }
      case 't': {
        num_threads = std::stoul(optarg);
        break;
      }
      case 'g': {
        use_gpu_buffer = parse_flag(optarg);
        break;
      }
      case 'd': {
        gpu_index = std::stoi(optarg);
        break;
      }
      case 'r': {
        repetition = std::stoi(optarg);
        break;
      }
      case 'w': {
        overwrite_file = parse_flag(optarg);
        break;
      }
      case LongOnlyOpts::O_DIRECT: {
        o_direct = parse_flag(optarg);
        break;
      }
      case LongOnlyOpts::ALIGN_BUFFER: {
        align_buffer = parse_flag(optarg);
        break;
      }
      case LongOnlyOpts::DROP_CACHE: {
        drop_file_cache = parse_flag(optarg);
        break;
      }
      case LongOnlyOpts::OPEN_ONCE: {
        open_file_once = parse_flag(optarg);
        break;
      }
      case 'h': {
        print_usage(argv[0]);
        std::exit(0);
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

  // Validation
  if (filepaths.empty()) { throw std::invalid_argument("--file is required"); }

  // Reset getopt state for second pass in the future
  optind = 1;
}

void Config::print_usage(std::string const& program_name)
{
  std::cout
    << "Usage: " << program_name << " [OPTIONS]\n\n"
    << "Options:\n"
    << "  -f, --file PATH                   File path to benchmark (required, repeatable)\n"
    << "  -s, --size SIZE                   Number of bytes to read (default: 4G)\n"
    << "                                    Supports suffixes: K, M, G, T, P\n"
    << "  -t, --threads NUM                 Number of threads (default: 1)\n"
    << "  -r, --repetitions NUM             Number of repetitions (default: 5)\n"
    << "  -g, --use-gpu-buffer BOOL         Use GPU device memory (default: false)\n"
    << "  -d, --gpu-index INDEX             GPU device index (default: 0)\n"
    << "  -w, --overwrite BOOL              Overwrite existing file (default: false)\n"
    << "  --o-direct BOOL                   Use O_DIRECT (default: true)\n"
    << "  --align-buffer BOOL               Use aligned buffer (default: true)\n"
    << "  --drop-cache BOOL                 Drop page cache before each run (default: false)\n"
    << "  --open-once BOOL                  Open file once, not per repetition (default: false)\n"
    << "  -h, --help                        Show this help message\n";
}

void* CudaPageAlignedDeviceAllocator::allocate(std::size_t size)
{
  void* buffer{};
  auto const page_size = get_page_size();
  auto const up_size   = size + page_size;
  KVIKIO_CHECK_CUDA(cudaMalloc(&buffer, up_size));
  auto* aligned_buffer = kvikio::detail::align_up(buffer, page_size);
  return aligned_buffer;
}

void CudaPageAlignedDeviceAllocator::deallocate(void* buffer, std::size_t /*size*/) {}

}  // namespace kvikio::benchmark
