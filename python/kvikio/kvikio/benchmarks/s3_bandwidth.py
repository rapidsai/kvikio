# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""
S3 read bandwidth benchmark for EC2.

Usage:
    python s3_bandwidth.py --backend easy_threadpool --no-gpu --submitters 48 \
        --window-seconds 60 --nthreads 480 \
        --url-file <(printf 's3://my-bucket/my-prefix/part.%d.parquet\n' {0..179})

    python s3_bandwidth.py --backend multi_poll --no-gpu --submitters 48 \
        --window-seconds 60 --num-reactors 48 --max-concurrent-requests 480 \
        --reactor-dispatch per_chunk \
        --url-file <(printf 's3://my-bucket/my-prefix/part.%d.parquet\n' {0..179})

--url-file takes one S3 URL per line. To list a bucket:

    aws s3 ls s3://my-bucket/my-prefix/ \
        | awk '{print "s3://my-bucket/my-prefix/" $4}' > urls.txt

How it measures:
Each submitter thread runs pread() then get() back to back, so --submitters reads are
in flight for the whole run. These are this script's own threads, distinct from
KvikIO's pool threads, which serve the range requests underneath and are sized by
--nthreads. Files are opened once up front. Bandwidth is taken over a window of
--window-seconds, which --warmup-seconds precedes rather than eats into, so the run
lasts the two added together.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time

import numpy as np

# ---------------------------------------------------------------------------
# Runtime settings
# ---------------------------------------------------------------------------

# Map each CLI option name to the KvikIO environment variable it sets.
_CLI_TO_ENV_VAR = {
    "nthreads": "KVIKIO_NTHREADS",
    "task_size": "KVIKIO_TASK_SIZE",
    "bounce_buffer_size": "KVIKIO_BOUNCE_BUFFER_SIZE",
    "num_reactors": "KVIKIO_REMOTE_IO_NUM_REACTORS",
    "reactor_dispatch": "KVIKIO_REMOTE_IO_REACTOR_DISPATCH",
    "max_concurrent_requests": "KVIKIO_REMOTE_IO_MAX_CONCURRENT_REQUESTS",
}

# Which backend reads each option. Anything absent here is read by both backends.
_ONLY_USED_BY = {
    "nthreads": "easy_threadpool",
    "num_reactors": "multi_poll",
    "reactor_dispatch": "multi_poll",
    "max_concurrent_requests": "multi_poll",
}

# Options the user must supply, keyed by the backend that needs them.
_REQUIRED_BY = {
    "nthreads": "easy_threadpool",
    "num_reactors": "multi_poll",
    "max_concurrent_requests": "multi_poll",
}

# Default task size.
_DEFAULT_TASK_SIZE = 64 * 1024**2

# KvikIO's own default, used to check --task-size against when the flag is not given.
_DEFAULT_BOUNCE_BUFFER_SIZE = 16 * 1024**2


# Map suffix to numerical multiplier. Lists longest suffix first. The order matters:
# "8mib" ends with "mib" and also with "b", and matching "b" first would leave "8mi" as
# the number. Keep any new suffix in length order.
_BYTE_UNITS = (
    ("kib", 1024),
    ("mib", 1024**2),
    ("gib", 1024**3),
    ("kb", 1000),
    ("mb", 1000**2),
    ("gb", 1000**3),
    ("b", 1),
)


def _parse_bytes(text: str) -> int:
    """Parse a byte count such as "8388608", "8MiB", or "8 MB"."""
    s = text.strip().lower().replace(" ", "").replace("_", "")

    # No suffix means a plain byte count, so start with a multiplier of 1.
    number, multiplier = s, 1
    for suffix, factor in _BYTE_UNITS:
        if s.endswith(suffix):
            number, multiplier = s[: -len(suffix)], factor
            break

    try:
        value = int(float(number) * multiplier)
    except (ValueError, OverflowError):
        raise argparse.ArgumentTypeError(f"invalid byte count {text!r}")
    if value <= 0:
        raise argparse.ArgumentTypeError(f"byte count must be positive: {text!r}")
    return value


def _validate_args(args) -> None:
    missing = []
    for option, backend in _REQUIRED_BY.items():
        if backend == args.backend and getattr(args, option) is None:
            missing.append("--" + option.replace("_", "-"))
    if missing:
        raise SystemExit(
            f"error: --backend {args.backend} requires " + ", ".join(sorted(missing))
        )
    if args.gpu and args.backend == "multi_poll":
        bounce = args.bounce_buffer_size or _DEFAULT_BOUNCE_BUFFER_SIZE
        if args.task_size > bounce:
            raise SystemExit(
                f"error: --backend multi_poll with --gpu requires --task-size "
                f"({args.task_size}) <= --bounce-buffer-size ({bounce})"
            )
    if args.submitters < 1:
        raise SystemExit("error: --submitters must be at least 1")
    if args.window_seconds <= 0:
        raise SystemExit("error: --window-seconds must be positive")
    if args.warmup_seconds < 0:
        raise SystemExit("error: --warmup-seconds must not be negative")


def _apply_settings(args) -> dict[str, str]:
    """Write the CLI settings into the environment, before KvikIO is imported.

    Returns the mapping that was written.
    """
    if "kvikio" in sys.modules:
        raise RuntimeError("runtime settings must be applied before kvikio is imported")

    env_vars = {"KVIKIO_REMOTE_IO_BACKEND": args.backend.upper()}

    for option, env_var in _CLI_TO_ENV_VAR.items():
        # Example:
        # option: "num_reactors"
        # cli_value: 48, or None if not passed by users
        cli_value = getattr(args, option)
        owning_backend = _ONLY_USED_BY.get(option)
        if owning_backend is not None and owning_backend != args.backend:
            if cli_value is not None:
                flag = "--" + option.replace("_", "-")
                print(
                    f"warning: {flag} has no effect under "
                    f"{args.backend.upper()}, ignoring it",
                    file=sys.stderr,
                )
            continue
        # An unset option is left out of the environment, so KvikIO keeps its own
        # default.
        if cli_value is not None:
            env_vars[env_var] = str(cli_value)

    if args.remote_verbose:
        env_vars["KVIKIO_REMOTE_VERBOSE"] = "1"
    env_vars["AWS_DEFAULT_REGION"] = args.aws_region

    os.environ.update(env_vars)
    return env_vars


def _verify_settings(env_vars: dict[str, str]) -> None:
    """Read the settings back from KvikIO to confirm they took effect."""
    import kvikio.defaults

    reported = kvikio.defaults.get("remote_io_backend").name
    expected = env_vars["KVIKIO_REMOTE_IO_BACKEND"]
    if reported != expected:
        raise RuntimeError(
            f"backend did not take effect: kvikio reports {reported}, "
            f"expected {expected}"
        )

    for env_var, config_name in (
        ("KVIKIO_NTHREADS", "num_threads"),
        ("KVIKIO_TASK_SIZE", "task_size"),
        ("KVIKIO_BOUNCE_BUFFER_SIZE", "bounce_buffer_size"),
    ):
        if env_var not in env_vars:
            continue
        reported_value = kvikio.defaults.get(config_name)
        if str(reported_value) != env_vars[env_var]:
            raise RuntimeError(
                f"{env_var} did not take effect: kvikio reports {reported_value}, "
                f"expected {env_vars[env_var]}"
            )


def _check_credentials() -> None:
    """Warn when the AWS environment looks incomplete."""
    # AWS_DEFAULT_REGION is not checked here. _apply_settings always sets it from
    # --aws-region, which has a default.
    required = ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY")
    missing = []
    for name in required:
        if not os.environ.get(name):
            missing.append(name)
    if missing:
        print(
            "warning: unset AWS variable(s): "
            + ", ".join(missing)
            + "\n         KvikIO reads credentials from the environment only, "
            "not from\n         ~/.aws, SSO caches, or EC2 instance metadata.",
            file=sys.stderr,
        )
    if os.environ.get("AWS_ACCESS_KEY_ID", "").startswith("ASIA"):
        if not os.environ.get("AWS_SESSION_TOKEN"):
            print(
                "warning: AWS_ACCESS_KEY_ID looks like an STS temporary key "
                "but AWS_SESSION_TOKEN is unset",
                file=sys.stderr,
            )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _allocate_buffer(nbytes: int, *, gpu: bool):
    """Return an nbytes uint8 destination buffer, on device or on host."""
    if gpu:
        import cupy as cp

        return cp.empty(nbytes, dtype=np.uint8)
    return np.empty(nbytes, dtype=np.uint8)


def _gbps(nbytes: int, seconds: float) -> float:
    """Convert a transfer to gigabits per second."""
    return nbytes * 8 / 1e9 / seconds


def _sleep_until(deadline: float) -> None:
    """Block until perf_counter() reaches deadline."""
    time.sleep(max(0.0, deadline - time.perf_counter()))


def _open_remote_files(urls: list[str]) -> list[tuple]:
    """Open every URL once. Used to keep the handle alive for the whole run.

    Returns a list of (handle, url, size_in_bytes).
    """
    import kvikio

    if not urls:
        raise ValueError("no URLs to read")

    out = []
    for url in urls:
        handle = kvikio.RemoteFile.open_s3_url(url)
        out.append((handle, url, handle.nbytes()))
    return out


def _build_work_items(files: list[tuple], read_size: int | None) -> list[tuple]:
    """Expand the open files into the units of work a submitter reads.

    Without read_size each file is fully read by a pread call. Otherwise the file is
    split into chunks of read_size and the remainder dropped. A file smaller than
    read_size is skipped.

    Returns a list of (file_index, file_offset, size).
    """
    items = []
    skipped = 0
    for file_index in range(len(files)):
        _handle, _url, size = files[file_index]
        if read_size is None:
            items.append((file_index, 0, size))
            continue
        if size < read_size:
            skipped += 1
            continue
        for slice_index in range(size // read_size):
            items.append((file_index, slice_index * read_size, read_size))

    if skipped > 0:
        print(
            f"warning: skipped {skipped} object(s) smaller than --read-size",
            file=sys.stderr,
        )
    if not items:
        raise SystemExit("error: no object is as large as --read-size")
    return items


class _WorkCounter:
    """Hands out the next work item index to whichever submitter asks first."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._value = 0

    def next(self) -> int:
        with self._lock:
            value = self._value
            self._value += 1
        return value


# ---------------------------------------------------------------------------
# Benchmark core
# ---------------------------------------------------------------------------


def _submitter(
    submitter_id: int,
    files: list[tuple],
    items: list[tuple],
    dst,
    counter: _WorkCounter,
    barrier: threading.Barrier,
    run_seconds: float,
    errors: list[tuple[int, BaseException]],
    gpu: bool,
    gpu_device: int,
) -> None:
    """Read one item at a time until the run deadline."""
    try:
        if gpu:
            import cupy as cp

            cp.cuda.Device(gpu_device).use()

        # Wait for every submitter plus the main thread.
        barrier.wait()
        t_stop = time.perf_counter() + run_seconds

        while time.perf_counter() < t_stop:
            file_index, file_offset, size = items[counter.next() % len(items)]
            handle = files[file_index][0]
            handle.pread(dst, size=size, file_offset=file_offset).get()
    except BaseException as exc:
        errors.append((submitter_id, exc))
        barrier.abort()


def run_bench(
    urls: list[str],
    *,
    gpu_device: int,
    gpu: bool,
    submitters: int,
    read_size: int | None,
    warmup_seconds: float,
    window_seconds: float,
    settings: dict[str, str] | None = None,
    json_out: bool = False,
) -> None:
    if submitters < 1:
        raise ValueError(f"submitters must be >= 1, got {submitters}")
    if warmup_seconds < 0:
        raise ValueError(f"warmup_seconds must be >= 0, got {warmup_seconds}")
    if settings is None:
        settings = {}

    import kvikio

    if gpu:
        import cupy as cp

        cp.cuda.Device(gpu_device).use()

    # Open every object once (untimed)
    t_open_start = time.perf_counter()
    files = _open_remote_files(urls)
    open_time = time.perf_counter() - t_open_start

    run_seconds = warmup_seconds + window_seconds

    items = _build_work_items(files, read_size)
    max_read = 0
    for _file_index, _file_offset, size in items:
        if size > max_read:
            max_read = size

    # Run header
    if settings:
        print("--> runtime settings")
        for env_var in sorted(settings):
            print(f"    {env_var:<40s}{settings[env_var]}")
        print()

    print("--> benchmark config")
    print(f"    objects         : {len(files)}")
    print(f"    work items      : {len(items)} of up to {max_read / 1024**3:.4f} GiB")
    if gpu:
        destination = f"gpu device {gpu_device}"
    else:
        destination = "host"
    print(f"    destination     : {destination}")
    print(f"    submitters      : {submitters} (reads held in flight)")
    print(
        f"    buffers         : {submitters} x {max_read / 1024**3:.4f} GiB = "
        f"{submitters * max_read / 1024**3:.4f} GiB"
    )
    print(f"    window          : {window_seconds} s (measured)")
    print(f"    warmup          : {warmup_seconds} s (excluded, runs first)")
    print(f"    run time        : {run_seconds} s (wall clock)")
    print(
        f"    open cost       : {open_time / len(files) * 1000:.4f} ms per object, "
        f"{open_time:.4f} s total (untimed, outside the loop)"
    )

    print()

    dsts = []
    for _ in range(submitters):
        buf = _allocate_buffer(max_read, gpu=gpu)
        buf.fill(0)
        dsts.append(buf)

    # Benchmark loop
    counter = _WorkCounter()
    barrier = threading.Barrier(submitters + 1)
    errors: list[tuple[int, BaseException]] = []
    threads = []
    monitor = kvikio.SummaryMonitor()
    for submitter_id in range(submitters):
        thread = threading.Thread(
            target=_submitter,
            args=(
                submitter_id,
                files,
                items,
                dsts[submitter_id],
                counter,
                barrier,
                run_seconds,
                errors,
                gpu,
                gpu_device,
            ),
            daemon=True,
        )
        threads.append(thread)
        thread.start()

    print(
        f"--> running {run_seconds} s ({warmup_seconds} warmup + "
        f"{window_seconds} measured)"
    )
    try:
        barrier.wait()
    except threading.BrokenBarrierError:
        monitor.stop()
        for thread in threads:
            thread.join()
        if errors:
            submitter_id, exc = errors[0]
            raise RuntimeError(f"submitter {submitter_id} failed: {exc}") from exc
        raise
    t_run_start = time.perf_counter()

    _sleep_until(t_run_start + warmup_seconds)
    baseline = monitor.get()
    proc_at_window_start = time.process_time()

    _sleep_until(t_run_start + warmup_seconds + window_seconds)
    window = monitor.since(baseline)
    proc_at_window_end = time.process_time()
    monitor.stop()

    for thread in threads:
        thread.join()

    for handle, _url, _size in files:
        handle.close()

    if errors:
        submitter_id, exc = errors[0]
        raise RuntimeError(f"submitter {submitter_id} failed: {exc}") from exc

    # Result
    print()
    window_wall = window.wall_ns / 1e9

    if window.num_reads == 0:
        print(
            "--> no reads completed inside the measurement window. "
            "Raise --window-seconds"
        )
        # Still emit JSON, so a sweep driver can tell an empty window from a crash.
        if json_out:
            print("JSON " + json.dumps({"settings": settings, "reads": 0}))
        return

    bw_gib = window.bytes_read / window_wall / 1024**3
    bw_gbps = _gbps(window.bytes_read, window_wall)
    cpu_cores = (proc_at_window_end - proc_at_window_start) / window_wall
    mean_read = window.mean_duration_ns / 1e9

    print(
        f"--> result  (window: {window.num_reads} reads, "
        f"{window.bytes_read / 1024**3:.4f} GiB in {window_wall:.4f} s)"
    )
    print(f"    bandwidth       : {bw_gib:.4f} GiB/s   {bw_gbps:.2f} Gbps")
    print(f"    mean read       : {mean_read:.4f} s")
    print(f"    cpu             : {cpu_cores:.2f} cores ({cpu_cores * 100:.1f} %)")

    short = window.bytes_requested - window.bytes_transferred
    if window.num_errors or short:
        print(
            f"warning: {window.num_errors} failed operation(s), {short} bytes short",
            file=sys.stderr,
        )

    if json_out:
        print(
            "JSON "
            + json.dumps(
                {
                    "settings": settings,
                    "gib_per_s": bw_gib,
                    "gbps": bw_gbps,
                    "cpu_cores": cpu_cores,
                    "bytes": window.bytes_read,
                    "seconds": window_wall,
                    "reads": window.num_reads,
                    "submitters": submitters,
                    "mean_read_s": mean_read,
                    "num_errors": window.num_errors,
                    "open_ms": open_time / len(files) * 1000,
                }
            )
        )


# ---------------------------------------------------------------------------
# Object list
# ---------------------------------------------------------------------------


def read_urls(path: str) -> list[str]:
    """Read the S3 URLs to benchmark, one per line.

    Blank lines and lines starting with # are ignored.
    """
    urls = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                urls.append(line)
    if not urls:
        raise SystemExit(f"error: no URLs found in {path}")
    return urls


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="S3 read bandwidth benchmark for EC2")

    runtime = parser.add_argument_group(
        "runtime settings",
        "Translated into KVIKIO_* environment variables before KvikIO is imported.",
    )
    runtime.add_argument(
        "--backend",
        choices=("easy_threadpool", "multi_poll"),
        required=True,
        help="Remote I/O backend (KVIKIO_REMOTE_IO_BACKEND)",
    )
    runtime.add_argument(
        "--nthreads",
        type=int,
        default=None,
        help="Thread pool size. EASY_THREADPOOL only, required (KVIKIO_NTHREADS)",
    )
    runtime.add_argument(
        "--task-size",
        type=_parse_bytes,
        default=_DEFAULT_TASK_SIZE,
        help="Size each read is split into, e.g. 8MiB (KVIKIO_TASK_SIZE) "
        "(default: 64MiB)",
    )
    runtime.add_argument(
        "--bounce-buffer-size",
        type=_parse_bytes,
        default=None,
        help="Staging buffer for device reads, at least --task-size "
        "(KVIKIO_BOUNCE_BUFFER_SIZE)",
    )
    runtime.add_argument(
        "--num-reactors",
        type=int,
        default=None,
        help="Reactor threads. MULTI_POLL only, required "
        "(KVIKIO_REMOTE_IO_NUM_REACTORS)",
    )
    runtime.add_argument(
        "--reactor-dispatch",
        choices=("per_chunk", "per_pread"),
        default=None,
        help="Dispatch policy. MULTI_POLL only "
        "(KVIKIO_REMOTE_IO_REACTOR_DISPATCH) (default: per_chunk)",
    )
    runtime.add_argument(
        "--max-concurrent-requests",
        type=int,
        default=None,
        help="In-flight range requests, 0 for unlimited. MULTI_POLL only, required "
        "(KVIKIO_REMOTE_IO_MAX_CONCURRENT_REQUESTS)",
    )
    runtime.add_argument(
        "--remote-verbose",
        action="store_true",
        help="Enable libcurl verbose output (KVIKIO_REMOTE_VERBOSE)",
    )
    runtime.add_argument(
        "--aws-region",
        default="us-east-2",
        help="AWS region (AWS_DEFAULT_REGION, default: %(default)s). Credentials "
        "come from the environment",
    )

    parser.add_argument(
        "-d",
        "--gpu-device",
        type=int,
        default=0,
        help="CUDA device ordinal (default: 0)",
    )
    parser.add_argument(
        "--gpu",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Read into a device buffer. --no-gpu measures the network path alone "
        "(default: --gpu)",
    )
    parser.add_argument(
        "--submitters",
        type=int,
        required=True,
        help="Threads submitting reads, one pread in flight each. Not KvikIO's pool, "
        "which --nthreads sizes",
    )
    parser.add_argument(
        "--read-size",
        type=_parse_bytes,
        default=None,
        help="Bytes per read, e.g. 256MiB. Smaller objects are skipped "
        "(default: whole objects)",
    )
    parser.add_argument(
        "--window-seconds",
        type=float,
        default=30.0,
        help="Seconds of measured window, excluding --warmup-seconds (default: 30)",
    )
    parser.add_argument(
        "--warmup-seconds",
        type=float,
        default=5.0,
        help="Seconds to run before the window opens, added to --window-seconds "
        "rather than taken from it. Set it above the slowest read (default: 5)",
    )
    parser.add_argument(
        "--url-file",
        required=True,
        help="File of S3 URLs, one per line. Blank lines and # comments ignored",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit a machine-readable 'JSON {...}' summary line",
    )
    args = parser.parse_args()

    _validate_args(args)

    settings = _apply_settings(args)
    _check_credentials()
    _verify_settings(settings)

    run_bench(
        read_urls(args.url_file),
        gpu_device=args.gpu_device,
        gpu=args.gpu,
        submitters=args.submitters,
        read_size=args.read_size,
        warmup_seconds=args.warmup_seconds,
        window_seconds=args.window_seconds,
        settings=settings,
        json_out=args.json,
    )
