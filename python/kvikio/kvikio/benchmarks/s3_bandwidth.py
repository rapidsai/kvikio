r"""
S3 read bandwidth benchmark for EC2.

Usage:
    python s3_bandwidth.py --backend easy_threadpool --no-gpu --batch-size 32 \
        --duration 60 --nthreads 480 \
        --url-file <(printf 's3://my-bucket/my-prefix/part.%d.parquet\n' {0..179})

    python s3_bandwidth.py --backend multi_poll --no-gpu --batch-size 32 \
        --duration 60 --num-reactors 48 --max-concurrent-requests 480 \
        --reactor-dispatch per_chunk \
        --url-file <(printf 's3://my-bucket/my-prefix/part.%d.parquet\n' {0..179})

--url-file takes one S3 URL per line. To list a bucket:

    aws s3 ls s3://my-bucket/my-prefix/ \
        | awk '{print "s3://my-bucket/my-prefix/" $4}' > urls.txt
"""

from __future__ import annotations

import argparse
import os
import sys
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

# Options the user must supply, keyed by the backend that needs them. Concurrency is the
# subject of this benchmark and KvikIO's defaults are single threaded, so an omitted
# value would produce a number that looks valid and measures the wrong thing.
_REQUIRED_BY = {
    "nthreads": "easy_threadpool",
    "num_reactors": "multi_poll",
    "max_concurrent_requests": "multi_poll",
}

# Default task size, the one tunable with a defensible value that does not depend on the
# instance.
_DEFAULT_TASK_SIZE = 64 * 1024**2


# Map suffix to numerical multiplier, listed longest suffix first. The order matters:
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


def _check_required_options(args) -> None:
    """Exit when an option the chosen run needs was not supplied.

    argparse cannot express "required only for this backend", so the check happens
    after parsing.
    """
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


def _apply_settings(args) -> dict[str, str]:
    """Translate the CLI settings into the environment KvikIO reads at startup. Done
    before KvikIO is imported.

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
    """Read the settings back from KvikIO to confirm they took effect.
    """
    import kvikio.defaults

    reported = kvikio.defaults.get("remote_io_backend").name
    expected = env_vars["KVIKIO_REMOTE_IO_BACKEND"]
    assert reported == expected, (
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
    """Convert a transfer to gigabits per second.
    """
    return nbytes * 8 / 1e9 / seconds


def _resolve_remote_files(urls: list[str]) -> list[tuple[str, int]]:
    """Open each URL once (untimed) to discover its size.

    Returns a list of (url, size_in_bytes).
    """
    import kvikio

    assert urls, "no URLs to read"

    out: list[tuple[str, int]] = []
    for url in urls:
        f = kvikio.RemoteFile.open_s3_url(url)
        try:
            sz = f.nbytes()
        finally:
            f.close()
        out.append((url, sz))
    return out


# ---------------------------------------------------------------------------
# Benchmark core
# ---------------------------------------------------------------------------


def run_bench(
    urls: list[str],
    *,
    gpu_device: int,
    gpu: bool,
    batch_size: int,
    warmup_batches: int,
    target_duration: float,
    settings: dict[str, str] | None = None,
    json_out: bool = False,
    quiet: bool = False,
) -> None:

    assert batch_size >= 1, f"batch_size must be >= 1, got {batch_size}"
    assert warmup_batches >= 0, f"warmup_batches must be >= 0, got {warmup_batches}"

    import kvikio

    if gpu:
        import cupy as cp

        cp.cuda.Device(gpu_device).use()

    # resolve files (untimed)
    files = _resolve_remote_files(urls)
    max_file_size = 0
    for _url, sz in files:
        if sz > max_file_size:
            max_file_size = sz

    if batch_size > len(files):
        print(
            f"warning: only {len(files)} object(s) available, capping "
            f"--batch-size from {batch_size} to {len(files)}",
            file=sys.stderr,
        )
        batch_size = len(files)

    dsts = []
    for _ in range(batch_size):
        dsts.append(_allocate_buffer(max_file_size, gpu=gpu))

    # run header
    if settings:
        print("--> runtime settings")
        for env_var in sorted(settings):
            print(f"    {env_var:<40s}{settings[env_var]}")
        print()
    print("--> benchmark config")
    print(f"    objects         : {len(files)}")
    print(f"    object size     : up to {max_file_size / 1024**3:.4f} GiB")
    if gpu:
        destination = f"gpu device {gpu_device}"
    else:
        destination = "host"
    print(f"    destination     : {destination}")
    print(f"    batch_size      : {batch_size}")
    print(f"    warmup_batches  : {warmup_batches}")
    print(f"    duration        : {target_duration} s (wall clock)")
    print()

    # benchmark loop
    steady_wall = 0.0
    steady_proc = 0.0
    steady_bytes = 0
    steady_count = 0
    steady_bws: list[float] = []

    open_time = 0.0
    open_count = 0

    batch_no = 0
    cursor = 0
    t_run_start = time.perf_counter()

    while time.perf_counter() - t_run_start < target_duration:
        batch = []
        for i in range(batch_size):
            batch.append(files[(cursor + i) % len(files)])
        cursor = (cursor + batch_size) % len(files)

        # untimed: open the batch
        handles = []
        t_open0 = time.perf_counter()
        for url, _sz in batch:
            handles.append(kvikio.RemoteFile.open_s3_url(url))
        open_time += time.perf_counter() - t_open0
        open_count += len(batch)

        # timed: submit every pread, then await them all.
        t0_wall = time.perf_counter()
        t0_proc = time.process_time()
        futures = []
        for i in range(len(batch)):
            _url, sz = batch[i]
            futures.append(handles[i].pread(dsts[i], size=sz))
        transferred = 0
        for fut in futures:
            transferred += fut.get()
        t1_proc = time.process_time()
        t1_wall = time.perf_counter()

        for fh in handles:
            fh.close()

        batch_bytes = 0
        for _url, sz in batch:
            batch_bytes += sz
        assert transferred == batch_bytes, (
            f"short read: got {transferred} bytes, expected {batch_bytes}"
        )

        batch_time = t1_wall - t0_wall
        batch_proc = t1_proc - t0_proc
        batch_bw = batch_bytes / batch_time / 1024**3

        batch_no += 1
        is_warmup = batch_no <= warmup_batches
        if not is_warmup:
            steady_bws.append(batch_bw)
            steady_wall += batch_time
            steady_proc += batch_proc
            steady_bytes += batch_bytes
            steady_count += 1

        if not quiet:
            tag = " [warmup]" if is_warmup else ""
            print(
                f"    [batch {batch_no}]{tag}  {batch_bytes / 1024**3:.4f} GiB "
                f"in {batch_time:.4f} s   {batch_bw:.4f} GiB/s   "
                f"{_gbps(batch_bytes, batch_time):.2f} Gbps"
            )

    # result
    print()
    if steady_count == 0 or steady_wall <= 0:
        print("--> no steady state batches. Raise --duration or lower "
              "--warmup-batches")
        return

    bw_gib = steady_bytes / steady_wall / 1024**3
    bw_gbps = _gbps(steady_bytes, steady_wall)
    cpu_pct = steady_proc / steady_wall * 100

    print(
        f"--> result  (steady state: {steady_count} batches, "
        f"{steady_bytes / 1024**3:.4f} GiB in {steady_wall:.4f} s)"
    )
    print(f"    bandwidth       : {bw_gib:.4f} GiB/s   {bw_gbps:.2f} Gbps")
    print(f"    cpu             : {cpu_pct:.2f} %")
    if open_count > 0:
        # Excluded from the timing above, so state it rather than hide it.
        print(
            f"    open cost       : {open_time / open_count * 1000:.4f} ms "
            f"per object, {open_time:.4f} s total (untimed)"
        )

    if json_out:
        import json as _json

        print("JSON " + _json.dumps({
            "settings": settings or {},
            "gib_per_s": bw_gib,
            "gbps": bw_gbps,
            "cpu_pct": cpu_pct,
            "bytes": steady_bytes,
            "seconds": steady_wall,
            "batches": steady_count,
            "batch_gib_per_s": steady_bws,
            "open_ms": open_time / open_count * 1000 if open_count else 0.0,
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
        "Translated into KVIKIO_* environment variables before KvikIO is "
        "imported. KvikIO reads each of them once at startup, so they cannot "
        "be changed while the benchmark runs.",
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
        help="Thread pool size, EASY_THREADPOOL only. Required for that backend, "
        "since it sets the read concurrency (KVIKIO_NTHREADS)",
    )
    runtime.add_argument(
        "--task-size",
        type=_parse_bytes,
        default=_DEFAULT_TASK_SIZE,
        help="Size each read is split into, accepts suffixes such as 8MiB "
        "(KVIKIO_TASK_SIZE) (default: 64MiB)",
    )
    runtime.add_argument(
        "--bounce-buffer-size",
        type=_parse_bytes,
        default=None,
        help="Size of each staging buffer for device reads, must be at least "
        "--task-size. Required with --gpu, ignored otherwise "
        "(KVIKIO_BOUNCE_BUFFER_SIZE)",
    )
    runtime.add_argument(
        "--num-reactors",
        type=int,
        default=None,
        help="Reactor threads, MULTI_POLL only. Required for that backend "
        "(KVIKIO_REMOTE_IO_NUM_REACTORS)",
    )
    runtime.add_argument(
        "--reactor-dispatch",
        choices=("per_chunk", "per_pread", "shared_queue"),
        default=None,
        help="Reactor dispatch policy, MULTI_POLL only. shared_queue needs a "
        "non-zero --max-concurrent-requests and falls back to per_chunk without "
        "one (KVIKIO_REMOTE_IO_REACTOR_DISPATCH) (default: KvikIO's per_chunk)",
    )
    runtime.add_argument(
        "--max-concurrent-requests",
        type=int,
        default=None,
        help="In-flight range requests across all reactors, MULTI_POLL only, "
        "0 means unlimited. Required for that backend "
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
        help="AWS region (AWS_DEFAULT_REGION, default: %(default)s). "
        "Credentials themselves are read from the environment and have no "
        "flag, so they stay out of the process command line",
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
        help="Read into a device buffer. Use --no-gpu to measure the network "
        "path alone, without the host-to-device hop (default: --gpu)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Objects read concurrently per batch (default: 1)",
    )
    parser.add_argument(
        "--warmup-batches",
        type=int,
        default=1,
        help="Leading batches excluded from the result (default: 1)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=30.0,
        help="Wall clock run time in seconds (default: 30)",
    )
    parser.add_argument(
        "--url-file",
        required=True,
        help="File listing the S3 URLs to read, one per line. Blank lines and "
        "lines starting with # are ignored. See the module docstring for ways to "
        "generate one",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress the per-batch lines, print only the result",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit a machine-readable 'JSON {...}' summary line",
    )
    args = parser.parse_args()

    _check_required_options(args)

    settings = _apply_settings(args)
    _check_credentials()
    _verify_settings(settings)

    run_bench(
        read_urls(args.url_file),
        gpu_device=args.gpu_device,
        gpu=args.gpu,
        batch_size=args.batch_size,
        warmup_batches=args.warmup_batches,
        target_duration=args.duration,
        settings=settings,
        json_out=args.json,
        quiet=args.quiet,
    )
