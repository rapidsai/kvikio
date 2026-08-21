# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
import json
import pickle
import time

import numpy as np
import pytest

import kvikio


def written_buffer(nbytes):
    """A buffer for a test that writes it out.

    Zeroed rather than `np.empty()`, or valgrind reports the uninitialised bytes reaching
    `pwrite()`, which they do.
    """
    return np.zeros(nbytes // 8, dtype="u8")


@pytest.fixture
def a_file(tmp_path):
    """A file with 1 MiB of data in it, and the number of bytes."""
    path = tmp_path / "test-file"
    data = np.arange(1024 * 1024 // 8, dtype="u8")
    with kvikio.CuFile(path, "w") as f:
        f.write(data)
    return path, data.nbytes


def test_counts_the_calls_it_spans(a_file, tmp_path):
    path, nbytes = a_file
    buffer = written_buffer(nbytes)

    monitor = kvikio.SummaryMonitor()
    with kvikio.CuFile(tmp_path / "written", "w") as f:
        f.write(buffer)
    with kvikio.CuFile(path, "r") as f:
        f.read(buffer)
        f.read(buffer)
    summary = monitor.get()

    assert summary.num_ops == 3
    assert summary.bytes_requested == 3 * nbytes
    assert summary.bytes_transferred == 3 * nbytes
    assert summary.num_errors == 0
    assert 0 < summary.busy_ns <= summary.wall_ns

    assert summary.num_reads == 2
    assert summary.num_writes == 1
    assert summary.num_reads + summary.num_writes == summary.num_ops
    assert summary.bytes_read == 2 * nbytes
    assert summary.bytes_written == nbytes
    assert summary.bytes_read + summary.bytes_written == summary.bytes_transferred


def test_a_failed_call_is_counted_as_an_error(a_file):
    path, nbytes = a_file
    buffer = written_buffer(nbytes)

    monitor = kvikio.SummaryMonitor()
    with pytest.raises(RuntimeError):
        with kvikio.CuFile(path, "r") as f:
            f.write(buffer)
    summary = monitor.get()

    assert summary.num_ops == 1
    assert summary.num_errors == 1
    assert summary.bytes_transferred == 0, "a failed call moved nothing"
    assert sum(b["num_errors"] for b in summary.by_backend.values()) == 1


def test_monitors_are_independent(a_file):
    path, nbytes = a_file
    buffer = np.empty(nbytes // 8, dtype="u8")

    outer = kvikio.SummaryMonitor()
    with kvikio.CuFile(path, "r") as f:
        f.read(buffer)
        inner = kvikio.SummaryMonitor()
        f.read(buffer)

    assert outer.get().num_ops == 2
    assert inner.get().num_ops == 1, "the read before the monitor was counted"

    before_reset = outer.get()
    outer.reset()
    after_reset = outer.get()
    assert after_reset.num_ops == 0
    assert after_reset.start_unix_ns > before_reset.start_unix_ns, (
        "the span did not restart"
    )
    assert inner.get().num_ops == 1, "resetting one must not touch the other"


def test_context_manager_stops_on_exit(a_file):
    path, nbytes = a_file
    buffer = np.empty(nbytes // 8, dtype="u8")

    with kvikio.SummaryMonitor() as monitor:
        with kvikio.CuFile(path, "r") as f:
            f.read(buffer)
    counted = monitor.get()
    assert counted.num_ops == 1

    with kvikio.CuFile(path, "r") as f:
        f.read(buffer)
    assert monitor.get().num_ops == 1, "counting continued after the block"
    monitor.stop()  # Idempotent, and the block already stopped it.
    assert monitor.get().num_ops == 1
    # The span ended with the block, so a later reading describes the same interval.
    assert monitor.get().end_unix_ns == counted.end_unix_ns
    assert monitor.get().wall_ns == counted.wall_ns


def test_a_summary_survives_a_round_trip_through_bytes(a_file):
    path, nbytes = a_file
    buffer = np.empty(nbytes // 8, dtype="u8")

    monitor = kvikio.SummaryMonitor()
    with kvikio.CuFile(path, "r") as f:
        f.read(buffer)
    summary = monitor.get()

    raw = summary.serialize()
    back = kvikio.Summary.deserialize(raw)
    assert back == summary
    # The report and the JSON come out the same, so the anchor survived too.
    assert str(back) == str(summary)
    assert back.to_json() == summary.to_json()
    # Still a reading of this monitor, so an interval can be measured from it.
    with kvikio.CuFile(path, "r") as f:
        f.read(buffer)
    assert monitor.since(back).num_ops == 1
    monitor.stop()

    # The C++ handle cannot be pickled, so both of these go through the bytes.
    assert pickle.loads(pickle.dumps(summary)) == summary
    assert copy.deepcopy(summary) == summary

    with pytest.raises(ValueError):
        kvikio.Summary.deserialize(raw[:-1])
    with pytest.raises(ValueError):
        kvikio.Summary.deserialize(b"not a summary")


def test_consecutive_intervals_partition_the_run(a_file):
    path, nbytes = a_file
    buffer = np.empty(nbytes // 8, dtype="u8")

    monitor = kvikio.SummaryMonitor()
    first = monitor.get()
    baseline = first
    intervals = []
    for _ in range(3):
        with kvikio.CuFile(path, "r") as f:
            f.read(buffer)
        now = monitor.get()
        intervals.append(now.since(baseline))
        baseline = now
    monitor.stop()

    # One interval describes its own tick, not the run so far.
    assert intervals[0].num_ops == 1
    assert intervals[0].bytes_transferred == nbytes
    assert intervals[0].busy_ns <= intervals[0].wall_ns

    # Nothing counted twice and nothing lost, in the counters or on the time axis.
    assert sum(i.num_ops for i in intervals) == 3
    assert sum(i.bytes_transferred for i in intervals) == 3 * nbytes
    assert sum(i.busy_ns for i in intervals) == baseline.busy_ns - first.busy_ns
    assert intervals[0].start_unix_ns == first.end_unix_ns
    for earlier, later in zip(intervals, intervals[1:]):
        assert later.start_unix_ns == earlier.end_unix_ns

    # An interval holds differences, so it cannot serve as the next baseline.
    with pytest.raises(ValueError):
        baseline.since(intervals[0])


def test_report_formats(a_file):
    path, nbytes = a_file
    buffer = np.empty(nbytes // 8, dtype="u8")

    monitor = kvikio.SummaryMonitor()
    with kvikio.CuFile(path, "r") as f:
        f.read(buffer)
    summary = monitor.get()

    text = str(summary)
    assert "KvikIO I/O summary" in text
    assert "bytes" in text

    parsed = json.loads(summary.to_json())
    assert parsed["num_ops"] == 1
    assert abs(time.time_ns() - parsed["start_unix_ns"]) < 60e9
    assert parsed["num_reads"] == 1
    assert parsed["bytes_transferred"] == nbytes
    assert parsed["bytes_read"] == nbytes
    assert parsed["total_duration_ns"] > 0
    assert set(parsed["by_backend"]) == set(summary.by_backend)
    assert len(summary.by_backend) == 5

    assert "num_ops=1" in repr(summary)


def test_derived_values(a_file):
    path, nbytes = a_file
    buffer = np.empty(nbytes // 8, dtype="u8")

    monitor = kvikio.SummaryMonitor()
    with kvikio.CuFile(path, "r") as f:
        for _ in range(4):
            f.read(buffer)
    summary = monitor.get()

    # Each read was waited on before the next, so nothing overlapped.
    assert summary.total_duration_ns > 0
    assert summary.total_duration_ns <= summary.busy_ns
    assert summary.mean_duration_ns == summary.total_duration_ns // summary.num_ops
    # Every operation belongs to exactly one backend, so the parts add up to the whole.
    per_backend = summary.by_backend.values()
    assert sum(b["num_ops"] for b in per_backend) == summary.num_ops
    assert sum(b["bytes_transferred"] for b in per_backend) == summary.bytes_transferred
    assert sum(b["total_duration_ns"] for b in per_backend) == summary.total_duration_ns
    assert sum(b["num_errors"] for b in per_backend) == summary.num_errors

    assert 0.0 < summary.busy_fraction <= 1.0
    assert summary.busy_bytes_per_sec > 0.0
    # Multiplying by the busy fraction recovers the whole-span rate.
    whole_span = summary.bytes_transferred * 1e9 / summary.wall_ns
    assert summary.busy_bytes_per_sec * summary.busy_fraction == pytest.approx(
        whole_span, rel=0.01
    )
