# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Convert KvikIO JSON physical-read logs into Chrome/Perfetto trace JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert KvikIO NDJSON read logs into Chrome/Perfetto trace format. "
            "Input is one JSON object per line."
        )
    )
    parser.add_argument("input", type=Path, help="Path to KvikIO read log NDJSON file")
    parser.add_argument("output", type=Path, help="Path to output trace JSON file")
    parser.add_argument(
        "--pid",
        type=int,
        default=1,
        help="Process id to use in trace events (default: 1)",
    )
    return parser.parse_args()


def _to_trace_event(
    record: dict[str, Any], tid_map: dict[int, int], pid: int
) -> dict[str, Any]:
    start_ns = int(record["start"])
    end_ns = int(record["end"])
    duration_ns = max(0, end_ns - start_ns)
    bytes_read = int(record.get("bytesRead", 0))
    throughput_bytes_per_second = (
        (bytes_read * 1_000_000_000.0 / duration_ns) if duration_ns > 0 else 0.0
    )

    raw_thread_id = int(record.get("threadId", 0))
    if raw_thread_id not in tid_map:
        tid_map[raw_thread_id] = len(tid_map) + 1
    tid = tid_map[raw_thread_id]

    args = {
        "source": record.get("source"),
        "offset": record.get("offset"),
        "size": record.get("size"),
        "bytesRead": bytes_read,
        "backend": record.get("backend"),
        "status": record.get("status"),
        "isDeviceBuffer": record.get("isDeviceBuffer"),
        "requestId": record.get("requestId"),
        "threadId": raw_thread_id,
        "startNs": start_ns,
        "endNs": end_ns,
        "durationNs": duration_ns,
        "throughputBytesPerSecond": throughput_bytes_per_second,
    }

    # Keep any future fields without dropping them.
    for key, value in record.items():
        if key not in args:
            args[key] = value

    return {
        "name": f"read:{record.get('backend', 'unknown')}",
        "cat": "kvikio.read",
        "ph": "X",
        "pid": pid,
        "tid": tid,
        "ts": start_ns / 1000.0,  # trace timestamps are in microseconds
        "dur": duration_ns / 1000.0,
        "args": args,
    }


def convert(input_path: Path, output_path: Path, pid: int) -> None:
    tid_map: dict[int, int] = {}
    trace_events: list[dict[str, Any]] = []

    with input_path.open() as f:
        for line_no, line in enumerate(f, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                record = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no}: {exc}") from exc

            # A JSON-formatted KvikIO log can contain ordinary messages as well
            # as physical-read events. Only reads map to duration events.
            if record.get("event", "read") != "read":
                continue

            missing = [field for field in ("start", "end") if field not in record]
            if missing:
                raise ValueError(
                    f"Missing required field(s) {missing!r} on line {line_no}: {text}"
                )

            trace_events.append(_to_trace_event(record, tid_map, pid))

    # Add thread metadata entries so normalized tids are easier to interpret.
    for original_tid, trace_tid in tid_map.items():
        trace_events.append(
            {
                "name": "thread_name",
                "ph": "M",
                "pid": pid,
                "tid": trace_tid,
                "args": {"name": f"kvikio-thread-{original_tid}"},
            }
        )

    output = {"displayTimeUnit": "ns", "traceEvents": trace_events}
    with output_path.open("w") as out:
        json.dump(output, out)


def main() -> None:
    args = _parse_args()
    convert(args.input, args.output, args.pid)


if __name__ == "__main__":
    main()
