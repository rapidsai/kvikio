/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <functional>
#include <span>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#include <kvikio/detail/transfer_plan.hpp>
#include <kvikio/error.hpp>
#include <kvikio/logger.hpp>

namespace kvikio::detail {

namespace {

// Requests sharing a handle and a CUDA context may merge, nothing else may. One transfer has one
// pinned buffer, one stream and one context, so a span mixing a host destination with a device one,
// or two device contexts, would produce a transfer that cannot complete. A null context marks a
// host destination, so the pair also encodes the memory kind with no invalid state.
struct GroupKey {
  RemoteHandle* handle;
  CUcontext cuda_context;

  bool operator==(GroupKey const& other) const noexcept = default;
};

struct GroupKeyHash {
  std::size_t operator()(GroupKey const& key) const noexcept
  {
    auto const h1 = std::hash<void const*>{}(key.handle);
    auto const h2 = std::hash<void const*>{}(key.cuda_context);
    return h1 ^ (h2 + 0x9e3779b97f4a7c15ULL + (h1 << 6) + (h1 >> 2));
  }
};

// A request's position in the sweep. Sorting these pairs rather than the requests themselves is
// what lets the input span stay const and keeps the caller's indices meaningful.
struct GroupEntry {
  std::size_t file_offset;
  std::size_t request_index;

  bool operator<(GroupEntry const& other) const noexcept
  {
    if (file_offset != other.file_offset) { return file_offset < other.file_offset; }
    return request_index < other.request_index;
  }
};

/**
 * @brief Emit the transfers for one run of requests, splitting the span at `task_size`.
 *
 * A run wider than `task_size` holds exactly one request, since merging stops at that cap. So
 * either the run has one request or it produces one transfer, and the nested loop below is linear
 * in the run size despite its shape.
 */
void emit_run(std::span<TransferPlanRequest const> requests,
              std::span<GroupEntry const> run,
              std::size_t run_offset,
              std::size_t run_end,
              std::size_t task_size,
              TransferPlan& plan)
{
  // Every request of a run shares the group key, so the first one speaks for all of them.
  auto const& run_head = requests[run.front().request_index];

  for (auto chunk_offset = run_offset; chunk_offset < run_end;) {
    auto const chunk_end     = std::min(chunk_offset + task_size, run_end);
    auto const segment_begin = plan.segments.size();
    std::size_t wanted_bytes = 0;

    for (auto const& entry : run) {
      auto const& req        = requests[entry.request_index];
      auto const piece_begin = std::max(req.file_offset, chunk_offset);
      auto const piece_end   = std::min(req.file_offset + req.size, chunk_end);
      if (piece_begin >= piece_end) { continue; }
      plan.segments.push_back({.span_offset = piece_begin - chunk_offset,
                               .length      = piece_end - piece_begin,
                               .dst = static_cast<char*>(req.dst) + (piece_begin - req.file_offset),
                               .request_index = entry.request_index});
      wanted_bytes += piece_end - piece_begin;
      ++plan.transfers_per_request[entry.request_index];
    }

    // A run starts and ends on a request boundary, and a chunk boundary can only fall inside a
    // single-request run, so no chunk is ever empty or starts inside a gap.
    assert(plan.segments.size() > segment_begin);
    assert(plan.segments[segment_begin].span_offset == 0);

    plan.transfers.push_back({.handle        = run_head.handle,
                              .cuda_context  = run_head.cuda_context,
                              .file_offset   = chunk_offset,
                              .size          = chunk_end - chunk_offset,
                              .segment_begin = segment_begin,
                              .segment_end   = plan.segments.size()});
    plan.overread_bytes += (chunk_end - chunk_offset) - wanted_bytes;
    chunk_offset = chunk_end;
  }
}

/**
 * @brief Sweep one group's sorted entries, growing runs greedily and emitting each as it closes.
 */
void plan_group(std::span<TransferPlanRequest const> requests,
                TransferPlanOptions const& opts,
                std::span<GroupEntry const> entries,
                TransferPlan& plan)
{
  std::size_t i = 0;
  while (i < entries.size()) {
    auto const first_entry = i;
    auto const& run_head   = requests[entries[i].request_index];
    auto const run_offset  = run_head.file_offset;
    auto run_end           = run_head.file_offset + run_head.size;
    ++i;

    while (opts.coalesce_max_gap.has_value() && i < entries.size()) {
      auto const& candidate = requests[entries[i].request_index];

      // Overlapping and duplicate ranges keep their own transfer. Redundant on the wire, but a
      // merged span cannot deliver the same byte to two destinations.
      if (candidate.file_offset < run_end) { break; }

      if (candidate.file_offset - run_end > *opts.coalesce_max_gap) { break; }

      // `task_size` doubles as the merge cap. Merging past it cannot bring the transfer count
      // below what splitting already achieves, and it would only add overread.
      if (candidate.file_offset + candidate.size - run_offset > opts.task_size) { break; }

      run_end = candidate.file_offset + candidate.size;
      ++i;
    }

    emit_run(requests,
             entries.subspan(first_entry, i - first_entry),
             run_offset,
             run_end,
             opts.task_size,
             plan);
  }
}

}  // namespace

TransferPlan build_transfer_plan(std::span<TransferPlanRequest const> requests,
                                 TransferPlanOptions const& opts)
{
  KVIKIO_EXPECT(opts.task_size > 0, "`task_size` must be positive", std::invalid_argument);

  TransferPlan plan;
  plan.transfers_per_request.assign(requests.size(), 0);
  if (requests.empty()) { return plan; }

  // Groups are kept in order of first appearance. Iterating the map instead would make the emitted
  // plan depend on where the allocator happened to place a handle.
  std::vector<std::vector<GroupEntry>> groups;
  std::unordered_map<GroupKey, std::size_t, GroupKeyHash> group_index;
  for (std::size_t i = 0; i < requests.size(); ++i) {
    auto const& req = requests[i];
    // Excluded rather than rejected, matching `pread()`, which returns a ready future for these.
    if (req.size == 0) { continue; }
    auto const [it, inserted] =
      group_index.try_emplace(GroupKey{req.handle, req.cuda_context}, groups.size());
    if (inserted) { groups.emplace_back(); }
    groups[it->second].push_back({.file_offset = req.file_offset, .request_index = i});
  }

  for (auto& entries : groups) {
    // Callers such as columnar readers usually hand over ascending ranges, and checking for that
    // is cheaper than sorting. No caller promise is needed, and none would save this pass.
    if (!std::is_sorted(entries.begin(), entries.end())) {
      std::sort(entries.begin(), entries.end());
    }
    plan_group(requests, opts, entries, plan);
  }

  // Overread is invisible otherwise, and without it there is no way to tune `coalesce_max_gap`.
  KVIKIO_LOG_DEBUG("build_transfer_plan(): %zu request(s) -> %zu transfer(s), %zu overread byte(s)",
                   requests.size(),
                   plan.transfers.size(),
                   plan.overread_bytes);

  return plan;
}

}  // namespace kvikio::detail
