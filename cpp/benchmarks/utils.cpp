/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "utils.hpp"

#include <getopt.h>

#include <algorithm>
#include <sstream>
#include <stdexcept>

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

  // Normalize to uppercase for case-insensitive comparison
  std::transform(
    suffix.begin(), suffix.end(), suffix.begin(), [](unsigned char c) { return std::tolower(c); });

  // All multipliers use 1024 (binary), not 1000
  std::size_t multiplier{1};

  // Support both K/Ki, M/Mi, etc. as synonyms (all 1024-based)
  std::size_t constexpr one_Ki{1024ULL};
  std::size_t constexpr one_Mi{1024ULL * one_Ki};
  std::size_t constexpr one_Gi{1024ULL * one_Mi};
  std::size_t constexpr one_Ti{1024ULL * one_Gi};
  if (suffix == "k" || suffix == "ki" || suffix == "kib") {
    multiplier = one_Ki;
  } else if (suffix == "m" || suffix == "mi" || suffix == "mib") {
    multiplier = one_Mi;
  } else if (suffix == "g" || suffix == "gi" || suffix == "gib") {
    multiplier = one_Gi;
  } else if (suffix == "t" || suffix == "ti" || suffix == "tib") {
    multiplier = one_Ti;
  } else {
    throw std::invalid_argument("Invalid size suffix: '" + suffix +
                                "' (use K/Ki/KiB, M/Mi/MiB, G/Gi/GiB, or T/Ti/TiB)");
  }

  return static_cast<std::size_t>(value * multiplier);
}

void Config::parse_args(int argc, char** argv)
{
  static option long_options[] = {
    {"file", required_argument, nullptr, 'f'},
    {"size", required_argument, nullptr, 's'},
    {"threads", required_argument, nullptr, 't'},
    {"repetitions", required_argument, nullptr, 'r'},
    {"no-direct", no_argument, nullptr, 'D'},
    {"no-align", no_argument, nullptr, 'A'},
    {"drop-cache", no_argument, nullptr, 'c'},
    // {"overwrite", no_argument, nullptr, 'w'},
    {"open-once", no_argument, nullptr, 'o'},
    {"help", no_argument, nullptr, 'h'},
    {0, 0, 0, 0}  // Sentinel to mark the end of the array. Needed by getopt_long()
  };

  int opt{0};
  int option_index{-1};

  // - By default getopt_long() returns '?' to indicate errors if an option has missing argument or
  // if an unknown option is encountered. The starting ':' in the optstring modifies this behavior.
  // Missing argument error now causes the return value to be ':'. Unknow option still leads to '?'
  // and its processing is deferred.
  // - "f:" means option "-f" takes an argument "c" means option
  // - "-c" does not take an argument
  while ((opt = getopt_long(argc, argv, ":f:s:t:r:DAcoh", long_options, &option_index)) != -1) {
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
      case 'r': {
        repetition = std::stoi(optarg);
        break;
      }
      case 'D': {
        o_direct = false;
        break;
      }
      case 'A': {
        align_buffer = false;
        break;
      }
      case 'c': {
        drop_file_cache = true;
        break;
      }
      case 'o': {
        open_file_once = true;
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
            << "  -o, --open-once         Open file once (not per iteration)\n"
            << "  -h, --help              Show this help message\n";
}

}  // namespace kvikio::benchmark
