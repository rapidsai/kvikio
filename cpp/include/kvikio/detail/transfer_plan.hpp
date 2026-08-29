/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cstddef>
#include <optional>
#include <span>
#include <vector>

#include <kvikio/shim/cuda.hpp>

namespace kvikio {
// The planner compares handle pointers and never dereferences them, so a forward declaration is
// enough.
class RemoteHandle;
}  // namespace kvikio

namespace kvikio::detail {

/**
 * @brief One read request as seen by the transfer planner.
 *
 * Deliberately distinct from the public batch-read request type. The CUDA context is resolved by
 * the caller rather than here, which keeps the planner free of CUDA calls and lets its tests run
 * without a CUDA context.
 */
struct TransferPlanRequest {
  RemoteHandle* handle{nullptr};    ///< Never dereferenced. Grouping uses pointer identity.
  CUcontext cuda_context{nullptr};  ///< Null exactly for host destinations.
  void* dst{nullptr};               ///< Start of the destination buffer for this request.
  std::size_t file_offset{0};       ///< Offset of the first requested byte in the remote file.
  std::size_t size{0};              ///< Zero-size requests are excluded from the plan.
};

/**
 * @brief Options controlling how requests are turned into transfers.
 *
 * Fields the public API exposes as optional are already resolved by the caller, so the planner
 * never consults `defaults::`.
 */
struct TransferPlanOptions {
  /**
   * @brief Maximum length of a single transfer's span, gap bytes included.
   *
   * Also the merge cap. Coalescing never produces a span longer than this, and a request larger
   * than `task_size` is split across as many transfers as it needs.
   */
  std::size_t task_size{0};

  /**
   * @brief Two ranges separated by at most this many unwanted bytes are merged into one transfer.
   *
   * The gap bytes are fetched and discarded, trading bandwidth for round trips. 0 merges only
   * exactly adjacent ranges. `std::nullopt` disables coalescing, so every request gets its own
   * transfer.
   */
  std::optional<std::size_t> coalesce_max_gap{std::nullopt};
};

/**
 * @brief One contiguous piece of a transfer's span that must land in a caller buffer.
 *
 * Segments record only wanted bytes. The holes between consecutive segments are the gaps that
 * coalescing pulled in, and they are discarded rather than represented. A piece of a span is
 * measured by `length`, while a whole request or transfer is measured by `size`.
 */
struct TransferSegment {
  std::size_t span_offset;    ///< Byte offset within the owning transfer's span. Add that
                              ///< transfer's `file_offset` for the absolute file offset.
  std::size_t length;         ///< Number of bytes to copy.
  void* dst;                  ///< Absolute address, already advanced when a request spans
                              ///< transfers. Unlike `span_offset`, this is not relative.
  std::size_t request_index;  ///< Index into the caller's request span.
};

/**
 * @brief One byte-range request to issue, plus the slice of segments it feeds.
 *
 * The contiguous file range it fetches is its span, the term the rest of this header uses. A span
 * covers whatever gap bytes coalescing pulled in, so it is at least as wide as the segments it
 * serves and never wider than `task_size`.
 */
struct PlannedTransfer {
  RemoteHandle* handle;       ///< Handle shared by every segment of this transfer.
  CUcontext cuda_context;     ///< Null exactly for host destinations.
  std::size_t file_offset;    ///< Offset of the span in the remote file.
  std::size_t size;           ///< Span length, gap bytes included. Never exceeds `task_size`.
  std::size_t segment_begin;  ///< Half-open range into `TransferPlan::segments`.
  std::size_t segment_end;
};

/**
 * @brief The transfers to issue for one batch of requests.
 *
 * Within a transfer, segments are sorted by `span_offset`, do not overlap, and cover both ends of
 * the span, so a span always begins and ends on wanted bytes.
 */
struct TransferPlan {
  std::vector<PlannedTransfer> transfers;
  std::vector<TransferSegment> segments;  ///< Flat. Referenced by half-open ranges.

  /**
   * @brief Number of transfers writing into each request, indexed like the caller's input.
   *
   * Sizes each request's completion countdown. Zero for zero-size requests. Derivable from
   * `segments`, but computed here once so callers do not repeat the pass.
   */
  std::vector<std::size_t> transfers_per_request;

  std::size_t overread_bytes{0};  ///< Total gap bytes that will be fetched and discarded.
};

/**
 * @brief Group, sort, merge and split read requests into the transfers that will serve them.
 *
 * Pure computation. No network, no libcurl and no CUDA calls.
 *
 * Requests are grouped by `(handle, cuda_context)` before anything else, and only requests of the
 * same group may merge. That is required for correctness rather than locality, because one
 * transfer has one pinned buffer, one stream and one context.
 *
 * The caller's span is never reordered. Sorting happens on internal `(file_offset, request_index)`
 * pairs, which is what keeps `transfers_per_request` and `TransferSegment::request_index` aligned
 * with the caller's indices.
 *
 * @param requests The requests to plan. Overlapping and duplicate ranges are never merged.
 * @param opts Planning options, with the caller's optional values already resolved.
 * @return The plan. Empty input yields an empty plan, which is not an error.
 * @exception std::invalid_argument if `opts.task_size` is zero.
 */
TransferPlan build_transfer_plan(std::span<TransferPlanRequest const> requests,
                                 TransferPlanOptions const& opts);

}  // namespace kvikio::detail
