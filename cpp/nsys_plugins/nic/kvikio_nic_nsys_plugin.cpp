/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Nsight Systems plugin that samples per-interface network bandwidth and emits it as NVTX counter
// groups, so the bandwidth curve lands on the same timeline as CUDA and KvikIO activity.
//
// nsys spawns this executable as a collector for the duration of a profiling session, enabled with
// `--enable=kvikio_nic[,args]` and discovered via `NSYS_PLUGIN_SEARCH_DIRS` (see nsys-plugin.yaml).
// It links only NVTX (header-only) and libdl, so it is standalone and does not depend on libkvikio.

#include <algorithm>
#include <charconv>
#include <chrono>
#include <csignal>
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
namespace constants = kvikio::nsys_plugin::constants;

namespace {

/**
 * @brief Tag type naming this plugin's NVTX domain.
 *
 * The domain is separate from libkvikio's own domain because the plugin runs as its own process; it
 * is created through the nvtx3 C++ registry (`nvtx3::domain::get`) rather than `nvtxDomainCreateA`.
 */
struct kvikio_nic_domain {
  static constexpr char const* name{"KvikIO NIC"};
};

// Set from a signal handler to request a clean shutdown. `volatile sig_atomic_t` is the only type
// an async-signal-safe handler may touch.
volatile std::sig_atomic_t g_stop = 0;

/**
 * @brief Parsed command line configuration.
 */
struct Config {
  std::chrono::microseconds interval{50000};  ///< Sampling interval.
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
               "  -i | --interval  Sampling interval in microseconds (default 50000)\n"
               "  -d | --device    Interface name regex (default: up or unknown, non-loopback)\n"
               "  -h | --help      Print this help message\n",
               prog);
  std::exit(code);
}

/**
 * @brief Parse the command line into a Config, exiting on `--help` or a malformed argument.
 *
 * Accepts `-i N` / `--interval N` / `--interval=N` (and the `-iN` short form) plus the equivalent
 * `-d` / `--device` forms, matching how nsys forwards `--enable=kvikio_nic,-d,eth0,-i,50000`.
 *
 * @param argc Argument count.
 * @param argv Argument vector.
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
        name = arg;
      } else {
        name         = arg.substr(0, eq);
        inline_value = arg.substr(eq + 1);
      }
    } else if (arg.size() >= 2 && arg.front() == '-') {
      name = arg.substr(0, 2);
      if (arg.size() > 2) { inline_value = arg.substr(2); }
    } else {
      std::fprintf(stderr,
                   "kvikio_nic: unexpected argument '%.*s'\n",
                   static_cast<int>(arg.size()),
                   arg.data());
      print_help_and_exit(argv[0], 2);
    }

    // Take this option's value from the inline form (`--interval=N`, `-iN`) or the next argument.
    auto take_value = [&]() -> std::string {
      if (inline_value.has_value()) { return std::string{inline_value.value()}; }
      if (i + 1 < argc) { return std::string{argv[++i]}; }
      std::fprintf(stderr,
                   "kvikio_nic: option '%.*s' requires a value\n",
                   static_cast<int>(name.size()),
                   name.data());
      print_help_and_exit(argv[0], 2);
    };

    if (name == "-h" || name == "--help") {
      print_help_and_exit(argv[0], 0);
    } else if (name == "-i" || name == "--interval") {
      auto const value        = take_value();
      long long parsed        = 0;
      auto const [ptr, ec]    = std::from_chars(value.data(), value.data() + value.size(), parsed);
      auto const fully_parsed = (ec == std::errc{} && ptr == value.data() + value.size());
      if (!fully_parsed || parsed <= 0) {
        std::fprintf(stderr,
                     "kvikio_nic: invalid interval '%s' (expected a positive integer)\n",
                     value.c_str());
        print_help_and_exit(argv[0], 2);
      }
      config.interval = std::chrono::microseconds{parsed};
    } else if (name == "-d" || name == "--device") {
      auto const value = take_value();
      try {
        config.device_filter = std::regex{value};
      } catch (std::regex_error const& e) {
        std::fprintf(
          stderr, "kvikio_nic: invalid device regex '%s' (%s)\n", value.c_str(), e.what());
        print_help_and_exit(argv[0], 2);
      }
    } else {
      std::fprintf(
        stderr, "kvikio_nic: unknown option '%.*s'\n", static_cast<int>(name.size()), name.data());
      print_help_and_exit(argv[0], 2);
    }
  }
  return config;
}

/**
 * @brief Choose the interfaces to monitor.
 *
 * @param config Parsed configuration.
 * @return With no `--device` filter, all up (or unknown) non-loopback interfaces. With a filter,
 * all interfaces whose name matches the regex, bypassing the up check so an explicit request is
 * honored. Sorted for a stable order.
 */
std::vector<std::string> select_interfaces(Config const& config)
{
  if (!config.device_filter.has_value()) { return default_interfaces(); }
  std::vector<std::string> result;
  std::error_code ec;
  std::filesystem::directory_iterator it{std::filesystem::path{constants::sysfs_net}, ec};
  if (ec) { return result; }
  for (auto const& entry : it) {
    auto name = entry.path().filename().string();
    if (std::regex_match(name, config.device_filter.value())) { result.push_back(std::move(name)); }
  }
  std::sort(result.begin(), result.end());
  return result;
}

/**
 * @brief Build the counter semantics that carry the rate unit for a whole counter group.
 *
 * Keeping the unit here (rather than in the schema field names) means the identifiers stay
 * unit-neutral and a future unit change touches only this value.
 *
 * @return A populated counter-semantics extension describing MiB/s values.
 */
nvtxSemanticsCounter_t make_rate_semantics()
{
  nvtxSemanticsCounter_t sem{};
  sem.header.structSize    = sizeof(nvtxSemanticsCounter_t);
  sem.header.semanticId    = NVTX_SEMANTIC_ID_COUNTERS_V1;
  sem.header.version       = NVTX_COUNTER_SEMANTIC_VERSION;
  sem.header.next          = nullptr;
  sem.flags                = NVTX_COUNTER_FLAGS_NONE;
  sem.unit                 = "MiB/s";
  sem.unitScaleNumerator   = 1;
  sem.unitScaleDenominator = 1;
  sem.limitType            = NVTX_COUNTER_LIMIT_UNDEFINED;
  return sem;
}

/**
 * @brief Register the unit-neutral {rx, tx} payload schema shared by every counter group.
 *
 * @param domain NVTX domain handle.
 * @return The schema id to pass as nvtxCounterAttr_t::schemaId.
 */
std::uint64_t register_rate_schema(nvtxDomainHandle_t domain)
{
  nvtxPayloadSchemaEntry_t const entries[] = {
    {.type = NVTX_PAYLOAD_ENTRY_TYPE_DOUBLE, .name = "rx"},
    {.type = NVTX_PAYLOAD_ENTRY_TYPE_DOUBLE, .name = "tx"},
  };
  nvtxPayloadSchemaAttr_t attr{};
  attr.fieldMask = NVTX_PAYLOAD_SCHEMA_ATTR_FIELD_TYPE | NVTX_PAYLOAD_SCHEMA_ATTR_FIELD_ENTRIES |
                   NVTX_PAYLOAD_SCHEMA_ATTR_FIELD_NUM_ENTRIES |
                   NVTX_PAYLOAD_SCHEMA_ATTR_FIELD_STATIC_SIZE;
  attr.type              = NVTX_PAYLOAD_SCHEMA_TYPE_STATIC;
  attr.entries           = entries;
  attr.numEntries        = 2;
  attr.payloadStaticSize = sizeof(NicRates);
  return nvtxPayloadSchemaRegister(domain, &attr);
}

/**
 * @brief Register an NVTX counter group in the given domain.
 *
 * @param domain NVTX domain handle.
 * @param name Counter group name (for example "nic_bandwidth.eth0").
 * @param schema_id Payload schema id from register_rate_schema.
 * @return The counter id to pass to nvtxCounterSample().
 */
std::uint64_t register_counter(nvtxDomainHandle_t domain,
                               std::string const& name,
                               std::uint64_t schema_id)
{
  // The semantics apply to the whole group and must outlive this call, so keep them static.
  static nvtxSemanticsCounter_t const rate_semantics = make_rate_semantics();
  nvtxCounterAttr_t attr{};
  attr.structSize = sizeof(nvtxCounterAttr_t);
  attr.schemaId   = schema_id;
  attr.name       = name.c_str();
  attr.scopeId    = NVTX_SCOPE_CURRENT_VM;
  attr.semantics  = &rate_semantics.header;
  attr.counterId  = NVTX_COUNTER_ID_NONE;
  return nvtxCounterRegister(domain, &attr);
}

}  // namespace

extern "C" {
// nsys stops the collector by sending SIGTERM (then SIGKILL after a grace period). Catch it so the
// loop breaks and the process exits cleanly with code 0 instead of an abnormal termination; SIGINT
// gives the same clean exit for standalone Ctrl-C runs.
static void kvikio_nic_handle_signal(int /*signum*/) { g_stop = 1; }
}

int main(int argc, char** argv)
{
  auto const config     = parse_args(argc, argv);
  auto const interfaces = select_interfaces(config);
  if (interfaces.empty()) {
    std::fprintf(stderr, "kvikio_nic: no matching network interfaces to monitor.\n");
    return 1;
  }

  std::signal(SIGTERM, kvikio_nic_handle_signal);
  std::signal(SIGINT, kvikio_nic_handle_signal);

  nvtxDomainHandle_t domain = nvtx3::domain::get<kvikio_nic_domain>();
  auto const schema_id      = register_rate_schema(domain);

  std::vector<std::uint64_t> counter_ids;
  counter_ids.reserve(interfaces.size());
  for (auto const& name : interfaces) {
    counter_ids.push_back(register_counter(domain, "nic_bandwidth." + name, schema_id));
  }
  auto const total_counter_id = register_counter(domain, "nic_bandwidth.total", schema_id);

  std::fprintf(stderr,
               "kvikio_nic: sampling %zu interface(s) every %lld us\n",
               interfaces.size(),
               static_cast<long long>(config.interval.count()));

  using clock = std::chrono::steady_clock;
  // The sysfs paths of the selected interfaces are precomputed once; each tick then reads only
  // those counters, independent of how many other interfaces exist on the host.
  NicCounterReader const reader{interfaces};
  auto prev   = reader.read();
  auto prev_t = clock::now();
  auto next_t = prev_t;

  while (g_stop == 0) {
    // Absolute deadlines keep the sampling frequency drift-free. A signal does not shorten the
    // sleep (libstdc++ restarts it over EINTR), so g_stop is observed at the next tick.
    next_t += config.interval;
    std::this_thread::sleep_until(next_t);

    auto const now  = clock::now();
    auto cur        = reader.read();
    double const dt = std::chrono::duration<double>{now - prev_t}.count();

    NicRates total{0.0, 0.0};
    for (std::size_t i = 0; i < interfaces.size(); ++i) {
      NicRates rates{0.0, 0.0};
      if (prev[i].has_value() && cur[i].has_value()) {
        rates = compute_rates(prev[i].value(), cur[i].value(), dt);
      }
      total.rx += rates.rx;
      total.tx += rates.tx;
      nvtxCounterSample(domain, counter_ids[i], &rates, sizeof(rates));
    }
    nvtxCounterSample(domain, total_counter_id, &total, sizeof(total));

    prev   = std::move(cur);
    prev_t = now;
  }

  return 0;
}
