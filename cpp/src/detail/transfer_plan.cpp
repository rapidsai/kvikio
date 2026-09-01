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

// Only requests sharing a handle and a CUDA context may merge. A null context means host.
struct GroupKey {
  RemoteHandle* handle;
  CUcontext cuda_context;
  bool operator==(GroupKey const& other) const noexcept = default;
};

struct GroupKeyHash {
  // Boost-style combine. The constant is the golden ratio, which scatters the low bits that two
  // heap pointers tend to share.
  std::size_t operator()(GroupKey const& key) const noexcept
  {
    auto const h1 = std::hash<void const*>{}(key.handle);
    auto const h2 = std::hash<void const*>{}(key.cuda_context);
    return h1 ^ (h2 + 0x9e3779b97f4a7c15ULL + (h1 << 6) + (h1 >> 2));
  }
};

/**
 * @brief Emit one transfer serving a run of two or more merged requests.
 *
 * `task_size` caps such a run, so it always fits one span and every request sits whole inside it.
 * No request needs trimming here.
 *
 * @param requests The caller's requests.
 * @param run_indices The run's requests, ascending by file offset and non-overlapping.
 * @param plan The plan to append to.
 */
void emit_merged_run(std::span<TransferPlanRequest const> requests,
                     std::span<std::size_t const> run_indices,
                     TransferPlan& plan)
{
  // The run is sorted and its requests do not overlap.
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

  // Everything the span covers beyond the requests is gap.
  plan.overread_bytes += (span_end - span_begin) - wanted_bytes;
}

/**
 * @brief Emit one transfer per `task_size` chunk of a single request. No overread in this case.
 *
 * @param requests The caller's requests.
 * @param request_index Index of the single request.
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
                             .dst           = static_cast<std::byte*>(request.dst) + into_request,
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
 * @brief Walk one group's sorted requests and append results to the plan.
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

      // No merge if candidate is an overlapping or duplicate range.
      if (candidate.file_offset < span_end) { break; }

      // No merge if the gap is too large.
      if (candidate.file_offset - span_end > opts.coalesce_max_gap.value()) { break; }

      // No merge if inclusion of candidate causes the size to exceed `task_size`.
      if (candidate.file_offset + candidate.size - span_begin > opts.task_size) { break; }

      // Now we can merge
      span_end = candidate.file_offset + candidate.size;
      ++i;
    }

    auto const run_indices = group_indices.subspan(run_begin, i - run_begin);
    if (run_indices.size() > 1) {
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

  // Example:
  // requests    R0(A,host)  R1(B,host)  R2(A,host)  R3(A,ctx1)  R4(B,host)  R5(A,host)
  //
  // group_slot        groups
  // (A, host) -> 0    groups[0] = { 0, 2, 5 }
  // (B, host) -> 1    groups[1] = { 1, 4 }
  // (A, ctx1) -> 2    groups[2] = { 3 }

  // Hash table iteration order depends on hashed pointers and varies per run. To make it stable,
  // here we iterate in insertion order.
  std::vector<std::vector<std::size_t>> groups;
  std::unordered_map<GroupKey, std::size_t, GroupKeyHash> group_slot;
  for (std::size_t request_index = 0; request_index < requests.size(); ++request_index) {
    auto const& request = requests[request_index];
    if (request.size == 0) { continue; }
    auto const [it, inserted] =
      group_slot.try_emplace(GroupKey{request.handle, request.cuda_context}, groups.size());
    if (inserted) { groups.emplace_back(); }
    groups[it->second].push_back(request_index);
  }

  auto const by_file_offset = [requests](std::size_t a, std::size_t b) {
    if (requests[a].file_offset != requests[b].file_offset) {
      return requests[a].file_offset < requests[b].file_offset;
    }
    return a < b;
  };

  for (auto& group_indices : groups) {
    if (!std::is_sorted(group_indices.begin(), group_indices.end(), by_file_offset)) {
      std::sort(group_indices.begin(), group_indices.end(), by_file_offset);
    }
    plan_group(requests, opts, group_indices, plan);
  }

  KVIKIO_LOG_DEBUG("build_transfer_plan(): %zu request(s) -> %zu transfer(s), %zu overread byte(s)",
                   requests.size(),
                   plan.transfers.size(),
                   plan.overread_bytes);

  return plan;
}

}  // namespace kvikio::detail
