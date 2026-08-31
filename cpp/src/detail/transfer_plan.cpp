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

/**
 * @brief Emit the one transfer that serves a run of two or more merged requests.
 *
 * Such a run never needs splitting, because `task_size` is the merge cap, so this emits exactly one
 * transfer and every request sits whole inside its span. That is why nothing here clips.
 *
 * @param requests The caller's requests.
 * @param run_indices Indices of the run's requests, ascending by file offset and non-overlapping.
 * @param plan The plan to append to.
 */
void emit_merged_run(std::span<TransferPlanRequest const> requests,
                     std::span<std::size_t const> run_indices,
                     TransferPlan& plan)
{
  // The run is sorted and its requests do not overlap, so the last one ends the span.
  auto const& first     = requests[run_indices.front()];
  auto const& last      = requests[run_indices.back()];
  auto const span_begin = first.file_offset;
  auto const span_end   = last.file_offset + last.size;

  auto const segment_begin = plan.segments.size();
  std::size_t wanted_bytes = 0;

  for (auto const request_index : run_indices) {
    auto const& request = requests[request_index];
    plan.segments.push_back({.span_offset   = request.file_offset - span_begin,
                             .length        = request.size,
                             .dst           = request.dst,
                             .request_index = request_index});
    wanted_bytes += request.size;
    ++plan.transfers_per_request[request_index];
  }

  plan.transfers.push_back({.handle        = first.handle,
                            .cuda_context  = first.cuda_context,
                            .file_offset   = span_begin,
                            .size          = span_end - span_begin,
                            .segment_begin = segment_begin,
                            .segment_end   = plan.segments.size()});

  // Whatever the span covers beyond the requests themselves is gap.
  plan.overread_bytes += (span_end - span_begin) - wanted_bytes;
}

/**
 * @brief Emit one transfer per `task_size` chunk of a single request.
 *
 * The only path that splits, and the only one a lone request takes. A request that already fits
 * yields one transfer, which is what every read looks like with coalescing off. There are no gaps
 * here, so this never adds overread, and each chunk is filled entirely by one segment.
 *
 * @param requests The caller's requests.
 * @param request_index Index of the lone request.
 * @param task_size Maximum transfer span.
 * @param plan The plan to append to.
 */
void emit_split_request(std::span<TransferPlanRequest const> requests,
                        std::size_t request_index,
                        std::size_t task_size,
                        TransferPlan& plan)
{
  auto const& request = requests[request_index];
  auto const file_end = request.file_offset + request.size;

  for (auto chunk_begin = request.file_offset; chunk_begin < file_end;) {
    auto const chunk_end     = std::min(chunk_begin + task_size, file_end);
    auto const segment_begin = plan.segments.size();
    auto const into_request  = chunk_begin - request.file_offset;

    plan.segments.push_back({.span_offset   = 0,
                             .length        = chunk_end - chunk_begin,
                             .dst           = static_cast<char*>(request.dst) + into_request,
                             .request_index = request_index});
    plan.transfers.push_back({.handle        = request.handle,
                              .cuda_context  = request.cuda_context,
                              .file_offset   = chunk_begin,
                              .size          = chunk_end - chunk_begin,
                              .segment_begin = segment_begin,
                              .segment_end   = plan.segments.size()});
    ++plan.transfers_per_request[request_index];
    chunk_begin = chunk_end;
  }
}

/**
 * @brief Sweep one group's sorted requests, growing runs greedily and emitting each as it closes.
 *
 * @param requests The caller's requests.
 * @param opts Planning options.
 * @param group_indices Indices of this group's requests, sorted by file offset.
 * @param plan The plan to append to.
 */
void plan_group(std::span<TransferPlanRequest const> requests,
                TransferPlanOptions const& opts,
                std::span<std::size_t const> group_indices,
                TransferPlan& plan)
{
  std::size_t i = 0;
  while (i < group_indices.size()) {
    auto const run_begin  = i;
    auto const& run_head  = requests[group_indices[i]];
    auto const span_begin = run_head.file_offset;
    auto span_end         = run_head.file_offset + run_head.size;
    ++i;

    while (opts.coalesce_max_gap.has_value() && i < group_indices.size()) {
      auto const& candidate = requests[group_indices[i]];

      // Overlapping and duplicate ranges keep their own transfer. Redundant on the wire, but a
      // merged span cannot deliver the same byte to two destinations.
      if (candidate.file_offset < span_end) { break; }

      if (candidate.file_offset - span_end > *opts.coalesce_max_gap) { break; }

      // `task_size` doubles as the merge cap. Merging past it cannot bring the transfer count
      // below what splitting already achieves, and it would only add overread.
      if (candidate.file_offset + candidate.size - span_begin > opts.task_size) { break; }

      span_end = candidate.file_offset + candidate.size;
      ++i;
    }

    // Only a lone request can exceed `task_size`, since the cap above bounds every longer run. So
    // merging and splitting never meet, and each case gets the simpler of the two emitters.
    auto const run_indices = group_indices.subspan(run_begin, i - run_begin);
    if (run_indices.size() > 1) {
      assert(span_end - span_begin <= opts.task_size);
      emit_merged_run(requests, run_indices, plan);
    } else {
      emit_split_request(requests, run_indices.front(), opts.task_size, plan);
    }
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

  // Each group holds the indices of its requests. Indices rather than copies of the requests is
  // what lets the caller's span stay const while we sort, which is what keeps every index in the
  // returned plan meaningful to the caller.
  //
  // Groups are kept in order of first appearance. Iterating the map instead would make the emitted
  // plan depend on where the allocator happened to place a handle.
  std::vector<std::vector<std::size_t>> groups;
  std::unordered_map<GroupKey, std::size_t, GroupKeyHash> group_index;
  for (std::size_t i = 0; i < requests.size(); ++i) {
    auto const& request = requests[i];
    // Excluded rather than rejected, matching `pread()`, which returns a ready future for these.
    if (request.size == 0) { continue; }
    auto const [it, inserted] =
      group_index.try_emplace(GroupKey{request.handle, request.cuda_context}, groups.size());
    if (inserted) { groups.emplace_back(); }
    groups[it->second].push_back(i);
  }

  // Ties are broken by index so that duplicate offsets keep the caller's order.
  auto const by_file_offset = [requests](std::size_t a, std::size_t b) {
    if (requests[a].file_offset != requests[b].file_offset) {
      return requests[a].file_offset < requests[b].file_offset;
    }
    return a < b;
  };

  for (auto& group : groups) {
    // Callers such as columnar readers usually hand over ascending ranges, and checking for that
    // is cheaper than sorting. No caller promise is needed, and none would save this pass.
    if (!std::is_sorted(group.begin(), group.end(), by_file_offset)) {
      std::sort(group.begin(), group.end(), by_file_offset);
    }
    plan_group(requests, opts, group, plan);
  }

  // Overread is invisible otherwise, and without it there is no way to tune `coalesce_max_gap`.
  KVIKIO_LOG_DEBUG("build_transfer_plan(): %zu request(s) -> %zu transfer(s), %zu overread byte(s)",
                   requests.size(),
                   plan.transfers.size(),
                   plan.overread_bytes);

  return plan;
}

}  // namespace kvikio::detail
