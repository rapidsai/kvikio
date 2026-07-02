/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <span>
#include <stdexcept>
#include <stop_token>
#include <string>
#include <string_view>
#include <thread>

#include <fcntl.h>
#include <unistd.h>

#include <nvtx3/nvToolsExtCounters.h>
#include <nvtx3/nvToolsExtPayload.h>
#include <nvtx3/nvtx3.hpp>

#include <kvikio/detail/nvtx.hpp>
#include <kvikio/error.hpp>
#include <kvikio/experimental/nic_monitor.hpp>

namespace kvikio::experimental {

namespace {

namespace constants {
constexpr double bytes_per_mib  = 1024.0 * 1024.0;
constexpr char const* sysfs_net = "/sys/class/net";
}  // namespace constants

/**
 * @brief Read a sysfs pseudo-file in a single read().
 *
 * @param path Path of the pseudo-file to read.
 * @param buf Scratch buffer that backs the returned view. At most `buf.size()` bytes are read, so
 * it never overflows.
 * @return A view of the bytes read (into @p buf), or std::nullopt if the file cannot be read. The
 * string view is not necessarily null terminated.
 */
std::optional<std::string_view> read_pseudo_file(char const* path, std::span<char> buf) noexcept
{
  auto const fd = ::open(path, O_RDONLY | O_CLOEXEC);
  if (fd < 0) { return std::nullopt; }
  auto const n = ::read(fd, buf.data(), buf.size());
  ::close(fd);
  if (n <= 0) { return std::nullopt; }
  return std::string_view{buf.data(), static_cast<std::size_t>(n)};
}

/**
 * @brief Parse a single unsigned integer from a sysfs pseudo-file.
 *
 * @param path Path of the pseudo-file to read.
 * @return The parsed value, or std::nullopt if the file is unparseable.
 */
std::optional<std::uint64_t> read_u64_file(std::string const& path) noexcept
{
  // A uint64 counter is at most 20 digits, so 32 bytes is ample (20 digits + newline)
  std::array<char, 32> buf;
  auto const content = read_pseudo_file(path.c_str(), buf);
  if (!content.has_value()) { return std::nullopt; }
  std::uint64_t value{};
  // from_chars is locale-independent, non-allocating, and non-throwing, and stops at the trailing
  // newline.
  [[maybe_unused]] auto const [_, ec] =
    std::from_chars(content->data(), content->data() + content->size(), value);
  if (ec != std::errc{}) { return std::nullopt; }
  return value;
}

/**
 * @brief Check whether a network interface is up.
 *
 * @param name Interface name.
 * @return True only when operstate is "up" or "unknown". False for any other state or if it cannot
 * be read.
 */
bool iface_is_up(std::string const& name)
{
  auto const path = std::string{constants::sysfs_net} + "/" + name + "/operstate";
  std::array<char, 16> buf;
  auto const content = read_pseudo_file(path.c_str(), buf);
  if (!content.has_value()) { return false; }
  std::string_view state = content.value();
  while (!state.empty() && (state.back() == '\n' || state.back() == ' ')) {
    state.remove_suffix(1);
  }
  // Some working NICs report "unknown" (loopback or virtual NICs), so this state should be
  // accepted. The other states (down, notpresent, lowerlayerdown, testing, dormant) should be
  // excluded.
  return state == "up" || state == "unknown";
}

/**
 * @brief Enumerate the default set of interfaces to monitor.
 *
 * @return All "up" non-loopback interfaces under `/sys/class/net`, sorted for a stable order.
 */
std::vector<std::string> default_interfaces()
{
  std::vector<std::string> result;
  std::error_code ec;
  std::filesystem::directory_iterator it{std::filesystem::path{constants::sysfs_net}, ec};
  if (ec) { return result; }
  for (auto const& entry : it) {
    auto name = entry.path().filename().string();
    if (name == "lo") { continue; }
    if (!iface_is_up(name)) { continue; }
    result.push_back(std::move(name));
  }
  std::sort(result.begin(), result.end());
  return result;
}

// One sample of a counter group: receive and transmit rates for one interface.
struct NicRates {
  double rx_mibps;
  double tx_mibps;
};

/**
 * @brief Register the {rx_MiBps, tx_MiBps} payload schema for the counter groups.
 *
 * @param domain NVTX domain handle.
 * @return The schema id to pass as nvtxCounterAttr_t::schemaId.
 */
std::uint64_t register_rate_schema(nvtxDomainHandle_t domain)
{
  nvtxPayloadSchemaEntry_t const entries[] = {
    {.type = NVTX_PAYLOAD_ENTRY_TYPE_DOUBLE, .name = "rx_MiBps"},
    {.type = NVTX_PAYLOAD_ENTRY_TYPE_DOUBLE, .name = "tx_MiBps"},
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
 * @brief Register an NVTX counter group in the given domain using a payload schema.
 *
 * @param domain NVTX domain handle.
 * @param name Counter name (for example "nic_MiBps.eth0").
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
  attr.scopeId    = NVTX_SCOPE_NONE;
  attr.counterId  = NVTX_COUNTER_ID_NONE;
  return nvtxCounterRegister(domain, &attr);
}

}  // namespace

std::map<std::string, NicCounters> read_nic_counters()
{
  std::map<std::string, NicCounters> result;
  std::error_code ec;
  std::filesystem::directory_iterator it{std::filesystem::path{constants::sysfs_net}, ec};
  if (ec) { return result; }
  for (auto const& entry : it) {
    auto const name  = entry.path().filename().string();
    auto const stats = entry.path() / "statistics";
    auto const rx    = read_u64_file((stats / "rx_bytes").string());
    auto const tx    = read_u64_file((stats / "tx_bytes").string());
    // If an interface exposes no parseable byte counters, skip it.
    if (!rx.has_value() || !tx.has_value()) { continue; }
    result.emplace(name, NicCounters{rx.value(), tx.value()});
  }
  return result;
}

NicBandwidthMonitor::NicBandwidthMonitor(double freq_hz, std::vector<std::string> interfaces)
  : _freq_hz{freq_hz}, _interfaces{std::move(interfaces)}
{
  KVIKIO_EXPECT(freq_hz > 0.0, "freq_hz must be positive", std::invalid_argument);
}

NicBandwidthMonitor::~NicBandwidthMonitor() { stop(); }

void NicBandwidthMonitor::start()
{
  if (_running) { return; }
  if (_interfaces.empty()) { _interfaces = default_interfaces(); }

  nvtxDomainHandle_t domain = nvtx3::domain::get<libkvikio_domain>();
  auto const schema_id      = register_rate_schema(domain);
  _counter_ids.clear();
  _counter_ids.reserve(_interfaces.size());
  for (auto const& name : _interfaces) {
    _counter_ids.push_back(register_counter(domain, "nic_MiBps." + name, schema_id));
  }
  _total_counter_id = register_counter(domain, "nic_MiBps.total", schema_id);

  _running = true;
  _thread  = std::jthread{[this](std::stop_token stop_token) { run(stop_token); }};
}

void NicBandwidthMonitor::stop()
{
  if (!_running) { return; }
  _thread.request_stop();
  if (_thread.joinable()) { _thread.join(); }
  _running = false;
}

bool NicBandwidthMonitor::running() const noexcept { return _running; }

std::vector<std::string> const& NicBandwidthMonitor::interfaces() const noexcept
{
  return _interfaces;
}

void NicBandwidthMonitor::run(std::stop_token stop_token)
{
  nvtxDomainHandle_t domain = nvtx3::domain::get<libkvikio_domain>();
  using clock               = std::chrono::steady_clock;
  auto const interval       = std::chrono::duration<double>{1.0 / _freq_hz};

  auto prev   = read_nic_counters();
  auto prev_t = clock::now();

  while (!stop_token.stop_requested()) {
    std::this_thread::sleep_for(std::chrono::duration_cast<clock::duration>(interval));

    auto const now  = clock::now();
    auto const cur  = read_nic_counters();
    double const dt = std::chrono::duration<double>{now - prev_t}.count();

    NicRates total{0.0, 0.0};
    for (std::size_t i = 0; i < _interfaces.size(); ++i) {
      NicRates rates{0.0, 0.0};
      auto const itc = cur.find(_interfaces[i]);
      auto const itp = prev.find(_interfaces[i]);
      // Guard the first tick and a missing interface; guard each direction's counter wrap/reset
      // independently.
      if (dt > 0.0 && itc != cur.end() && itp != prev.end()) {
        if (itc->second.rx_bytes >= itp->second.rx_bytes) {
          auto const delta = itc->second.rx_bytes - itp->second.rx_bytes;
          rates.rx_mibps   = static_cast<double>(delta) / dt / constants::bytes_per_mib;
        }
        if (itc->second.tx_bytes >= itp->second.tx_bytes) {
          auto const delta = itc->second.tx_bytes - itp->second.tx_bytes;
          rates.tx_mibps   = static_cast<double>(delta) / dt / constants::bytes_per_mib;
        }
      }
      total.rx_mibps += rates.rx_mibps;
      total.tx_mibps += rates.tx_mibps;
      nvtxCounterSample(domain, _counter_ids[i], &rates, sizeof(rates));
    }
    nvtxCounterSample(domain, _total_counter_id, &total, sizeof(total));

    prev   = cur;
    prev_t = now;
  }
}

}  // namespace kvikio::experimental
