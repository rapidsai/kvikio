/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Nsight Systems plugin that samples per-interface network bandwidth and emits it as NVTX counter
// groups, so the bandwidth curve lands on the same timeline with the application activity.
//
// nsys spawns this executable as a collector for the duration of a profiling session, enabled with
// `--enable=kvikio_nic[,args]` and discovered via `NSYS_PLUGIN_SEARCH_DIRS`. It depends only on
// NVTX headers and libdl, so it is standalone and does not depend on libkvikio.

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <csignal>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <optional>
#include <regex>
#include <string>
#include <string_view>
#include <system_error>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#include <nvtx3/nvToolsExtCounters.h>
#include <nvtx3/nvToolsExtPayload.h>
#include <nvtx3/nvToolsExtSemanticsCounters.h>
#include <nvtx3/nvtx3.hpp>

#include "nic_monitor.hpp"

using kvikio::nsys_plugin::compute_rates;
using kvikio::nsys_plugin::default_interfaces;
using kvikio::nsys_plugin::NicCounterReader;
using kvikio::nsys_plugin::NicRates;

namespace {

/**
 * @brief Tag type naming this plugin's NVTX domain.
 *
 * The domain is separate from libkvikio's own domain because the plugin runs its own process.
 */
struct kvikio_nic_domain {
  static constexpr char const* name{"KvikIO NIC"};
};

// Set from a signal handler for a clean shutdown.
volatile std::sig_atomic_t g_stop = 0;

namespace constants {
// Exit codes
constexpr int exit_success       = 0;
constexpr int exit_no_interfaces = 1;
constexpr int exit_usage_error   = 2;

constexpr nvtxSemanticsCounter_t rate_semantics{
  .header               = {.structSize = sizeof(nvtxSemanticsCounter_t),
                           .semanticId = NVTX_SEMANTIC_ID_COUNTERS_V1,
                           .version    = NVTX_COUNTER_SEMANTIC_VERSION,
                           .next       = nullptr},
  .flags                = NVTX_COUNTER_FLAGS_NONE,
  .unit                 = "MiB/s",
  .unitScaleNumerator   = 1,
  .unitScaleDenominator = 1,
  .limitType            = NVTX_COUNTER_LIMIT_UNDEFINED,
};
}  // namespace constants

/**
 * @brief Parsed command line configuration.
 */
struct Config {
  std::chrono::microseconds interval{20000};  ///< Sampling interval.
  std::optional<std::regex> device_filter;    ///< If set, monitor interfaces matching this regex.
};

/**
 * @brief Print usage to stderr and terminate the process.
 *
 * @param prog Program name (argv[0]).
 * @param code Process exit code.
 */
[[noreturn]] void print_help_and_exit(char const* prog, int code)
{
  std::fprintf(stderr,
               "Usage: %s [options]\n"
               "  -i | --interval  Sampling interval in microseconds (default 20000)\n"
               "  -d | --device    Interface name regex (default: up or unknown, non-loopback)\n"
               "  -h | --help      Print this help message\n",
               prog);
  std::exit(code);
}

/**
 * @brief Parse the command line into a Config, exiting on `--help` or a malformed argument.
 *
 * Accepts arguments of these forms to be more consistent with nsys built-in sample:
 * - `-i N`
 * - `--interval N`
 * - `--interval=N`
 * - `-iN`
 *
 * Note that on the nsys command line the plugin arguments follow the plugin name as a comma
 * separated list (commas only, no spaces). These are therefore equivalent:
 * - `--enable=kvikio_nic,-d,eth0,-i,20000`
 * - `--enable=kvikio_nic,--device=eth0,--interval=20000`
 *
 * @param argc Program argument count.
 * @param argv Program argument vector.
 * @return The parsed configuration.
 */
Config parse_args(int argc, char** argv)
{
  Config config;
  for (int i = 1; i < argc; ++i) {
    std::string_view const arg{argv[i]};
    std::string_view name;
    std::optional<std::string_view> inline_value;
    if (arg.starts_with("--")) {
      auto const eq = arg.find('=');
      if (eq == std::string_view::npos) {
        // Example: --interval 20000
        name = arg;  // --interval
      } else {
        // Example: --interval=20000
        name         = arg.substr(0, eq);   // --interval
        inline_value = arg.substr(eq + 1);  // 20000
      }
    } else if (arg.size() >= 2 && arg.front() == '-') {
      // Example: -i 20000 or -i20000
      name = arg.substr(0, 2);  // -i
      if (arg.size() > 2) {
        // Example: -i20000
        inline_value = arg.substr(2);  // 20000
      }
    } else {
      std::fprintf(stderr,
                   "kvikio_nic: unexpected argument '%.*s'\n",
                   static_cast<int>(arg.size()),
                   arg.data());
      print_help_and_exit(argv[0], constants::exit_usage_error);
    }

    // Take this option's value from the inline form (`--interval=N`, `-iN`) or the next argument.
    auto take_value = [&]() -> std::string {
      if (inline_value.has_value()) { return std::string{inline_value.value()}; }
      if (i + 1 < argc) { return std::string{argv[++i]}; }
      std::fprintf(stderr,
                   "kvikio_nic: option '%.*s' requires a value\n",
                   static_cast<int>(name.size()),
                   name.data());
      print_help_and_exit(argv[0], constants::exit_usage_error);
    };

    if (name == "-h" || name == "--help") {
      print_help_and_exit(argv[0], constants::exit_success);
    } else if (name == "-i" || name == "--interval") {
      auto const value        = take_value();
      long long parsed        = 0;
      auto const [ptr, ec]    = std::from_chars(value.data(), value.data() + value.size(), parsed);
      auto const fully_parsed = (ec == std::errc{} && ptr == value.data() + value.size());
      if (!fully_parsed || parsed <= 0) {
        std::fprintf(stderr,
                     "kvikio_nic: invalid interval '%s' (expected a positive integer)\n",
                     value.c_str());
        print_help_and_exit(argv[0], constants::exit_usage_error);
      }
      config.interval = std::chrono::microseconds{parsed};
    } else if (name == "-d" || name == "--device") {
      auto const value = take_value();
      try {
        config.device_filter = std::regex{value};
      } catch (std::regex_error const& e) {
        std::fprintf(
          stderr, "kvikio_nic: invalid device regex '%s' (%s)\n", value.c_str(), e.what());
        print_help_and_exit(argv[0], constants::exit_usage_error);
      }
    } else {
      std::fprintf(
        stderr, "kvikio_nic: unknown option '%.*s'\n", static_cast<int>(name.size()), name.data());
      print_help_and_exit(argv[0], constants::exit_usage_error);
    }
  }
  return config;
}

/**
 * @brief Choose the interfaces to monitor.
 *
 * @param config Parsed configuration.
 * @return With no `--device` filter, all up or unknown, non-loopback interfaces. With a filter, all
 * interfaces whose name matches the regex, bypassing the up check so an explicit request is
 * honored. Sorted for a stable order.
 */
std::vector<std::string> select_interfaces(Config const& config)
{
  if (!config.device_filter.has_value()) { return default_interfaces(); }
  std::vector<std::string> result;
  std::error_code ec;
  std::filesystem::directory_iterator it{
    std::filesystem::path{kvikio::nsys_plugin::constants::sysfs_net}, ec};
  if (ec) { return result; }
  for (auto const& entry : it) {
    auto name = entry.path().filename().string();
    if (std::regex_match(name, config.device_filter.value())) { result.push_back(std::move(name)); }
  }
  std::sort(result.begin(), result.end());
  return result;
}

/**
 * @brief Register the {rx, tx} payload schema.
 *
 * @param domain NVTX domain handle.
 * @return The schema id to pass as nvtxCounterAttr_t::schemaId.
 */
std::uint64_t register_rate_schema(nvtxDomainHandle_t domain)
{
  static_assert(std::is_standard_layout_v<NicRates>);
  std::array const entries = {
    nvtxPayloadSchemaEntry_t{.type        = NVTX_PAYLOAD_ENTRY_TYPE_DOUBLE,
                             .name        = "rx",
                             .description = "Receive rate",
                             .offset      = offsetof(NicRates, rx)},
    nvtxPayloadSchemaEntry_t{.type        = NVTX_PAYLOAD_ENTRY_TYPE_DOUBLE,
                             .name        = "tx",
                             .description = "Transmit rate",
                             .offset      = offsetof(NicRates, tx)},
  };
  nvtxPayloadSchemaAttr_t attr{};
  attr.fieldMask = NVTX_PAYLOAD_SCHEMA_ATTR_FIELD_NAME | NVTX_PAYLOAD_SCHEMA_ATTR_FIELD_TYPE |
                   NVTX_PAYLOAD_SCHEMA_ATTR_FIELD_FLAGS | NVTX_PAYLOAD_SCHEMA_ATTR_FIELD_ENTRIES |
                   NVTX_PAYLOAD_SCHEMA_ATTR_FIELD_NUM_ENTRIES |
                   NVTX_PAYLOAD_SCHEMA_ATTR_FIELD_STATIC_SIZE;
  attr.name              = "NIC bandwidth";
  attr.type              = NVTX_PAYLOAD_SCHEMA_TYPE_STATIC;
  attr.flags             = NVTX_PAYLOAD_SCHEMA_FLAG_COUNTER_GROUP;
  attr.entries           = entries.data();
  attr.numEntries        = entries.size();
  attr.payloadStaticSize = sizeof(NicRates);
  return nvtxPayloadSchemaRegister(domain, &attr);
}

/**
 * @brief Register an NVTX counter group in the given domain.
 *
 * @param domain NVTX domain handle.
 * @param name Counter group name (for example "eth0").
 * @param schema_id Payload schema id from register_rate_schema.
 * @return The counter id to pass to nvtxCounterSample().
 */
std::uint64_t register_counter(nvtxDomainHandle_t domain,
                               std::string const& name,
                               std::uint64_t schema_id)
{
  nvtxCounterAttr_t attr{};
  attr.structSize = sizeof(nvtxCounterAttr_t);
  attr.schemaId   = schema_id;
  attr.name       = name.c_str();
  attr.scopeId    = NVTX_SCOPE_CURRENT_VM;
  attr.semantics  = &constants::rate_semantics.header;
  attr.counterId  = NVTX_COUNTER_ID_NONE;
  return nvtxCounterRegister(domain, &attr);
}

/**
 * @brief Samples NIC byte counters at a fixed interval and emits them as NVTX counter groups.
 *
 * Construction registers one `<iface>` counter group per interface plus `total`. `run()` then
 * differences the counters each tick until the stop flag is raised by the signal handler.
 */
class BandwidthCollector {
 public:
  /**
   * @brief Register the NVTX schema and counter groups for the given interfaces.
   *
   * @param interfaces Interfaces to monitor.
   * @param interval Sampling interval.
   */
  BandwidthCollector(std::vector<std::string> interfaces, std::chrono::microseconds interval)
    : _interval{interval},
      _reader{std::move(interfaces)},
      _domain{nvtx3::domain::get<kvikio_nic_domain>()}
  {
    auto const schema_id = register_rate_schema(_domain);
    _counter_ids.reserve(_reader.interfaces().size());
    for (auto const& name : _reader.interfaces()) {
      _counter_ids.push_back(register_counter(_domain, name, schema_id));
    }
    _total_counter_id = register_counter(_domain, "total", schema_id);
  }

  /**
   * @brief Run the sampling loop until @p stop becomes nonzero.
   *
   * @param stop Stop flag, set asynchronously by the signal handler.
   */
  void run(volatile std::sig_atomic_t const& stop) const
  {
    auto const& interfaces = _reader.interfaces();
    std::fprintf(stderr,
                 "kvikio_nic: sampling %zu interface(s) every %lld us\n",
                 interfaces.size(),
                 static_cast<long long>(_interval.count()));

    using clock        = std::chrono::steady_clock;
    auto prev_counters = _reader.read();
    auto prev_time     = clock::now();
    auto next_deadline = prev_time;

    while (stop == 0) {
      // Use sleep_until (instead of sleep_for) to minimize sampling frequency drift.
      // A signal does not interrupt the sleep.
      next_deadline += _interval;
      std::this_thread::sleep_until(next_deadline);

      auto const now    = clock::now();
      auto cur_counters = _reader.read();
      auto const dt     = std::chrono::duration<double>{now - prev_time}.count();

      NicRates total{0.0, 0.0};
      for (std::size_t i = 0; i < interfaces.size(); ++i) {
        NicRates rates{0.0, 0.0};
        if (prev_counters[i].has_value() && cur_counters[i].has_value()) {
          rates = compute_rates(prev_counters[i].value(), cur_counters[i].value(), dt);
        }
        total.rx += rates.rx;
        total.tx += rates.tx;
        nvtxCounterSample(_domain, _counter_ids[i], &rates, sizeof(rates));
      }
      nvtxCounterSample(_domain, _total_counter_id, &total, sizeof(total));

      prev_counters = std::move(cur_counters);
      prev_time     = now;
    }
  }

 private:
  std::chrono::microseconds _interval;
  NicCounterReader _reader;
  nvtxDomainHandle_t _domain;
  std::vector<std::uint64_t> _counter_ids;
  std::uint64_t _total_counter_id{0};
};

}  // namespace

// C language linkage (extern) to be standard conformant.
// Also internal linkage (static) as a good practice.
// nsys stops the collector by sending SIGTERM (then SIGKILL after a grace period). Catch it so the
// loop breaks and the process exits cleanly with code 0 instead of an abnormal termination. Also
// catch SIGINT to have the same clean exit for Ctrl-C.
extern "C" {
static void kvikio_nic_handle_signal(int /*signum*/) { g_stop = 1; }
}

int main(int argc, char** argv)
{
  auto const config = parse_args(argc, argv);
  auto interfaces   = select_interfaces(config);
  if (interfaces.empty()) {
    std::fprintf(stderr, "kvikio_nic: no matching network interfaces to monitor.\n");
    return constants::exit_no_interfaces;
  }

  std::signal(SIGTERM, kvikio_nic_handle_signal);
  std::signal(SIGINT, kvikio_nic_handle_signal);

  BandwidthCollector const collector{std::move(interfaces), config.interval};
  collector.run(g_stop);
  return constants::exit_success;
}
