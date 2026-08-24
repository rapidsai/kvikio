r"""
S3 read bandwidth benchmark for EC2.

Usage:
    python s3_bandwidth.py --backend easy_threadpool --no-gpu --workers 48 \
        --duration 60 --nthreads 480 \
        --url-file <(printf 's3://my-bucket/my-prefix/part.%d.parquet\n' {0..179})

    python s3_bandwidth.py --backend multi_poll --no-gpu --workers 48 \
        --duration 60 --num-reactors 48 --max-concurrent-requests 480 \
        --reactor-dispatch per_chunk \
        --url-file <(printf 's3://my-bucket/my-prefix/part.%d.parquet\n' {0..179})

--url-file takes one S3 URL per line. To list a bucket:

    aws s3 ls s3://my-bucket/my-prefix/ \
        | awk '{print "s3://my-bucket/my-prefix/" $4}' > urls.txt

How it measures:
Each worker runs pread() then get() back to back, so exactly --workers reads are in
flight for the whole run. Files are opened once up front, since open() is a HEAD
request and measures latency rather than bandwidth. Bandwidth is taken over a window
that drops the leading --warmup-seconds.
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
    """Exit on any argument combination argparse cannot express itself."""
    missing = []
    for option, backend in _REQUIRED_BY.items():
        if backend == args.backend and getattr(args, option) is None:
            missing.append("--" + option.replace("_", "-"))
    if missing:
        raise SystemExit(
            f"error: --backend {args.backend} requires " + ", ".join(sorted(missing))
        )
    if args.gpu and args.bounce_buffer_size is None:
        raise SystemExit("error: --bounce-buffer-size is required with --gpu")
    if args.warmup_seconds >= args.duration:
        raise SystemExit(
            "error: --warmup-seconds must be less than --duration, otherwise the "
            "measurement window is empty"
        )


def _apply_settings(args) -> dict[str, str]:
    """Write the CLI settings into the environment, before KvikIO is imported.

    Returns the mapping that was written.
    """
    assert "kvikio" not in sys.modules, (
        "runtime settings must be applied before kvikio is imported"
    )

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
    assert reported == expected, (
        f"backend did not take effect: kvikio reports {reported}, expected {expected}"
    )

    for env_var, config_name in (
        ("KVIKIO_NTHREADS", "num_threads"),
        ("KVIKIO_TASK_SIZE", "task_size"),
        ("KVIKIO_BOUNCE_BUFFER_SIZE", "bounce_buffer_size"),
    ):
        if env_var not in env_vars:
            continue
        reported_value = kvikio.defaults.get(config_name)
        assert str(reported_value) == env_vars[env_var], (
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
    """Block until perf_counter() reaches deadline. sleep() may return early."""
    while True:
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            return
        time.sleep(remaining)


def _open_remote_files(urls: list[str]) -> list[tuple]:
    """Open every URL once and keep the handle for the whole run.

    A RemoteFile pins no socket: pread() creates a curl handle per sub-range.

    Returns a list of (handle, url, size_in_bytes).
    """
    import kvikio

    assert urls, "no URLs to read"

    out = []
    for url in urls:
        handle = kvikio.RemoteFile.open_s3_url(url)
        out.append((handle, url, handle.nbytes()))
    return out


def _build_work_items(files: list[tuple], read_size: int | None) -> list[tuple]:
    """Expand the open files into the units of work a worker reads.

    Without read_size each item is a whole object. Otherwise objects are cut into
    whole slices of read_size and the remainder dropped, so every read is the same
    size. An object smaller than read_size is skipped rather than read whole, since
    one short read would smear the latency distribution.

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

    if skipped:
        print(
            f"warning: skipped {skipped} object(s) smaller than --read-size",
            file=sys.stderr,
        )
    if not items:
        raise SystemExit("error: no object is as large as --read-size")
    return items


class _WorkCounter:
    """Hands out the next work item index to whichever worker asks first.

    A plain lock is enough: one call per read, and a read takes seconds.
    """

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


def _worker(
    worker_id: int,
    files: list[tuple],
    items: list[tuple],
    dst,
    counter: _WorkCounter,
    barrier: threading.Barrier,
    target_duration: float,
    records: list,
    errors: list,
    gpu: bool,
    gpu_device: int,
) -> None:
    """Read one item at a time until the run deadline, recording every read.

    One pread() and get() per iteration, so this thread holds exactly one read in
    flight. `records` is this worker's own list and needs no lock.

    Workers may share a handle. That is safe: the endpoint is read-only after
    construction and every sub-range gets its own curl handle.
    """
    try:
        if gpu:
            import cupy as cp

            # CuPy tracks the current device per thread, so each worker sets its own.
            cp.cuda.Device(gpu_device).use()

        # Wait for every worker plus the main thread, so the run starts at full
        # concurrency instead of ramping up over thread creation.
        barrier.wait()
        t_stop = time.perf_counter() + target_duration

        while time.perf_counter() < t_stop:
            file_index, file_offset, size = items[counter.next() % len(items)]
            handle = files[file_index][0]

            t_read_start = time.perf_counter()
            future = handle.pread(dst, size=size, file_offset=file_offset)
            transferred = future.get()
            t_read_end = time.perf_counter()

            assert transferred == size, (
                f"short read: got {transferred} bytes, expected {size}"
            )
            records.append((t_read_start, t_read_end, transferred))
    except BaseException as exc:  # noqa: BLE001
        errors.append((worker_id, exc))
        # A worker that dies before the barrier would hang every other worker on it.
        barrier.abort()


def run_bench(
    urls: list[str],
    *,
    gpu_device: int,
    gpu: bool,
    workers: int,
    read_size: int | None,
    warmup_seconds: float,
    target_duration: float,
    settings: dict[str, str] | None = None,
    json_out: bool = False,
) -> None:
    assert workers >= 1, f"workers must be >= 1, got {workers}"
    assert warmup_seconds >= 0, f"warmup_seconds must be >= 0, got {warmup_seconds}"
    if settings is None:
        settings = {}

    if gpu:
        import cupy as cp

        cp.cuda.Device(gpu_device).use()

    # Open every object once (untimed)
    t_open_start = time.perf_counter()
    files = _open_remote_files(urls)
    open_time = time.perf_counter() - t_open_start

    items = _build_work_items(files, read_size)
    max_read = 0
    for _file_index, _file_offset, size in items:
        if size > max_read:
            max_read = size

    # Workers may share an item, so there is no reason to cap them at the item count.

    # One buffer per worker, sized to the largest read. Filling it forces the page
    # faults now, so the first reads of the run do not pay for them.
    dsts = []
    for _ in range(workers):
        buf = _allocate_buffer(max_read, gpu=gpu)
        buf.fill(0)
        dsts.append(buf)

    # Run header
    if settings:
        print("--> runtime settings")
        for env_var in sorted(settings):
            print(f"    {env_var:<40s}{settings[env_var]}")
        print()

    task_size = int(settings.get("KVIKIO_TASK_SIZE", _DEFAULT_TASK_SIZE))
    subranges_per_read = (max_read + task_size - 1) // task_size
    nominal_subranges = workers * subranges_per_read

    print("--> benchmark config")
    print(f"    objects         : {len(files)}")
    print(f"    work items      : {len(items)} of up to {max_read / 1024**3:.4f} GiB")
    if gpu:
        destination = f"gpu device {gpu_device}"
    else:
        destination = "host"
    print(f"    destination     : {destination}")
    print(f"    workers         : {workers} (reads held in flight)")
    print(f"    subranges/read  : {subranges_per_read} (read size / task size)")
    print(f"    subranges live  : {nominal_subranges} (nominal)")
    print(f"    buffers         : {workers} x {max_read / 1024**3:.4f} GiB")
    print(f"    duration        : {target_duration} s (wall clock)")
    print(f"    window          : {warmup_seconds} s warmup")
    print(
        f"    open cost       : {open_time / len(files) * 1000:.4f} ms per object, "
        f"{open_time:.4f} s total (untimed, outside the loop)"
    )

    # Below the backend's cap it starves, well above it the surplus only queues.
    cap_env = (
        "KVIKIO_NTHREADS"
        if settings.get("KVIKIO_REMOTE_IO_BACKEND") == "EASY_THREADPOOL"
        else "KVIKIO_REMOTE_IO_MAX_CONCURRENT_REQUESTS"
    )
    cap = int(settings.get(cap_env, 0))
    if cap > 0:
        print(f"    backend cap     : {cap} ({cap_env})")
        if nominal_subranges < cap:
            print(
                f"warning: {nominal_subranges} subranges in flight cannot fill a cap "
                f"of {cap}. Raise --workers to at least "
                f"{-(-cap // subranges_per_read)}",
                file=sys.stderr,
            )
    print()

    # Benchmark loop
    counter = _WorkCounter()
    barrier = threading.Barrier(workers + 1)
    records = []
    errors = []
    threads = []
    for worker_id in range(workers):
        records.append([])
        thread = threading.Thread(
            target=_worker,
            args=(
                worker_id,
                files,
                items,
                dsts[worker_id],
                counter,
                barrier,
                target_duration,
                records[worker_id],
                errors,
                gpu,
                gpu_device,
            ),
            daemon=True,
        )
        threads.append(thread)
        thread.start()

    print(f"--> running {target_duration} s")
    try:
        barrier.wait()
    except threading.BrokenBarrierError:
        for thread in threads:
            thread.join()
        if errors:
            worker_id, exc = errors[0]
            raise RuntimeError(f"worker {worker_id} failed: {exc}") from exc
        raise
    t_run_start = time.perf_counter()
    window_start = t_run_start + warmup_seconds
    # No trim at the end: workers only stop *starting* reads at the deadline, so
    # concurrency is still full when the window closes.
    window_end = t_run_start + target_duration

    # Sample CPU at the window edges so the figure covers the measured window only.
    _sleep_until(window_start)
    proc_at_window_start = time.process_time()
    _sleep_until(window_end)
    proc_at_window_end = time.process_time()

    for thread in threads:
        thread.join()

    for handle, _url, _size in files:
        handle.close()

    if errors:
        worker_id, exc = errors[0]
        raise RuntimeError(f"worker {worker_id} failed: {exc}") from exc

    all_records = []
    for per_worker in records:
        all_records.extend(per_worker)

    # Result
    print()
    window_wall = window_end - window_start

    # A read counts if it *finished* inside the window, with all of its bytes. That
    # over-counts reads straddling the start and under-counts those straddling the
    # end. Both edges hold the same number of half-done reads, so the two errors are
    # equal in expectation: unbiased over repeats, but a sub-percent residual per run.
    window_bytes = 0
    latencies = []
    for t_read_start, t_read_end, nbytes in all_records:
        if window_start <= t_read_end <= window_end:
            window_bytes += nbytes
            latencies.append(t_read_end - t_read_start)

    if not latencies:
        print("--> no reads completed inside the measurement window. Raise --duration")
        # Still emit JSON, so a sweep driver can tell an empty window from a crash.
        if json_out:
            print("JSON " + json.dumps({"settings": settings, "reads": 0}))
        return

    bw_gib = window_bytes / window_wall / 1024**3
    bw_gbps = _gbps(window_bytes, window_wall)
    cpu_cores = (proc_at_window_end - proc_at_window_start) / window_wall
    lat = np.array(latencies)
    p50, p90, p99 = np.percentile(lat, [50, 90, 99])

    print(
        f"--> result  (window: {len(latencies)} reads, "
        f"{window_bytes / 1024**3:.4f} GiB in {window_wall:.4f} s)"
    )
    print(f"    bandwidth       : {bw_gib:.4f} GiB/s   {bw_gbps:.2f} Gbps")
    print(
        f"    read latency    : p50 {p50:.4f} s   p90 {p90:.4f} s   p99 {p99:.4f} s"
    )
    print(f"    cpu             : {cpu_cores:.2f} cores ({cpu_cores * 100:.1f} %)")

    if json_out:
        print("JSON " + json.dumps({
            "settings": settings,
            "gib_per_s": bw_gib,
            "gbps": bw_gbps,
            "cpu_cores": cpu_cores,
            "bytes": window_bytes,
            "seconds": window_wall,
            "reads": len(latencies),
            "workers": workers,
            "latency_p50": float(p50),
            "latency_p90": float(p90),
            "latency_p99": float(p99),
            "open_ms": open_time / len(files) * 1000,
        }))


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
    parser = argparse.ArgumentParser(
        description="S3 read bandwidth benchmark for EC2")

    runtime = parser.add_argument_group(
        "runtime settings",
        "Translated into KVIKIO_* environment variables before KvikIO is imported."
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
        help="Staging buffer for device reads, at least --task-size. Required with "
        "--gpu (KVIKIO_BOUNCE_BUFFER_SIZE)",
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
        choices=("per_chunk", "per_pread", "shared_queue"),
        default=None,
        help="Dispatch policy. MULTI_POLL only; shared_queue needs a non-zero "
        "--max-concurrent-requests (KVIKIO_REMOTE_IO_REACTOR_DISPATCH) "
        "(default: per_chunk)",
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
        help="AWS region (AWS_DEFAULT_REGION, default: %(default)s). Credentials are "
        "read from the environment and have no flag",
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
        "--workers",
        type=int,
        required=True,
        help="Worker threads, each holding one pread in flight."
    )
    parser.add_argument(
        "--read-size",
        type=_parse_bytes,
        default=None,
        help="Bytes per read, e.g. 256MiB. Also sets how many range requests one "
        "read splits into, ceil(read size / --task-size) (default: whole objects)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=30.0,
        help="Wall clock run time in seconds (default: 30)",
    )
    parser.add_argument(
        "--warmup-seconds",
        type=float,
        default=5.0,
        help="Leading seconds excluded from the measurement window. Set it above the "
        "p99 read latency: a read that started cold still counts if it finishes "
        "inside the window (default: 5)",
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
        workers=args.workers,
        read_size=args.read_size,
        warmup_seconds=args.warmup_seconds,
        target_duration=args.duration,
        settings=settings,
        json_out=args.json,
    )
