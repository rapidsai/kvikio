"""
KvikIO S3 remote read microbenchmark.

Reads a hardcoded list of S3 URLs via ``kvikio.RemoteFile.open_s3_url`` and
measures aggregate read bandwidth.

Time-based: accumulates only the read wall time and iterates until the
cumulative read time exceeds a target duration.  If time remains after
reading all files, the file list is cycled from the start; per-pass and
overall aggregate bandwidths are reported separately.

Supports batched submission: with ``--batch-size N`` the benchmark groups
files into batches of N, submits all preads in a batch concurrently, then
awaits all completions before moving on.  Each batch uses one destination
buffer per slot (all sized to the largest file).  Default batch size 1
preserves single-file serial behaviour.

The first ``--warmup-batches M`` batches of the run (counted across pass
boundaries) are classified as warmup; the remainder are steady state.
End-of-run summary reports warmup, steady state, and overall aggregates
separately.

Reported metrics per segment:
    - bandwidth (GiB/s)
    - cpu utilisation (%)  -- ratio of process_time to wall time, expressed
      as a percentage; sum of user + system CPU time across all threads of
      this process (including kvikio's C++ worker threads).  Values may
      exceed 100% on multi-threaded workloads, matching the convention of
      ``top``, ``pidstat``, and ``/usr/bin/time -v``.
    - cpu% / (GiB/s)  -- CPU cost per unit throughput, equivalent to
      ``process_time / bytes`` (scaled).  Lower is better.

S3 credentials and KvikIO tunables (chunk size, thread pool, etc.) are
configured externally via the usual AWS and ``KVIKIO_*`` environment
variables -- this script does not touch them.

Usage:
    # Edit REMOTE_FILES below, then:
    python kvikio_remote_read_bench.py
    python kvikio_remote_read_bench.py --duration 60 -d 3
    python kvikio_remote_read_bench.py --batch-size 4 --warmup-batches 8
    python kvikio_remote_read_bench.py --no-gpu --no-align-buffer
"""

from __future__ import annotations

import argparse
import mmap
import os
import time

import numpy as np

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_PAGE_SIZE = os.sysconf("SC_PAGE_SIZE")


def _allocate_buffer(nbytes: int, *, gpu: bool, aligned: bool):
    """Return a (buffer, keepalive) tuple of *nbytes* uint8 bytes.

    GPU + aligned:    Over-allocate by one page, slice to page boundary.
    GPU + unaligned:  Plain cupy.empty (cudaMalloc, 256-byte aligned).
    Host + aligned:   mmap (page-aligned).
    Host + unaligned: Plain numpy.empty.
    """
    if gpu:
        import cupy as cp

        if not aligned:
            return cp.empty(nbytes, dtype=np.uint8), None

        backing = cp.cuda.alloc(nbytes + _PAGE_SIZE)
        ptr = backing.ptr
        misalign = ptr % _PAGE_SIZE
        offset = (_PAGE_SIZE - misalign) % _PAGE_SIZE
        aligned_ptr = ptr + offset

        mem = cp.cuda.UnownedMemory(aligned_ptr, nbytes, owner=backing)
        memptr = cp.cuda.MemoryPointer(mem, 0)
        buf = cp.ndarray(nbytes, dtype=np.uint8, memptr=memptr)

        assert buf.data.ptr % _PAGE_SIZE == 0, (
            f"device buffer not page-aligned: ptr=0x{buf.data.ptr:x}"
        )
        return buf, backing

    if aligned:
        backing = mmap.mmap(-1, nbytes)
        return np.frombuffer(backing, dtype=np.uint8), None

    return np.empty(nbytes, dtype=np.uint8), None


def _resolve_remote_files(urls: list[str]) -> list[tuple[str, int]]:
    """Open each URL once (untimed) to discover its size.

    Returns a list of (url, size_in_bytes).
    """
    import kvikio

    assert urls, "REMOTE_FILES is empty"

    out: list[tuple[str, int]] = []
    for url in urls:
        f = kvikio.RemoteFile.open_s3_url(url)
        try:
            sz = f.nbytes()
        finally:
            f.close()
        out.append((url, sz))
    return out


def _percentiles(bws: list[float]) -> str:
    """Format the batch-bandwidth distribution for one segment.

    Aggregate bandwidth alone hides the behaviour that separates the remote
    backends: they reach a similar peak, and differ in how often a batch falls
    well short of it.  Reporting the low percentiles makes that visible.
    """
    if not bws:
        return ""
    s = sorted(bws)

    def pct(p: float) -> float:
        # Nearest-rank on the sorted sample; p is the fraction *below* which
        # the value falls, so p10 is a slow batch and p90 a fast one.
        idx = min(len(s) - 1, max(0, int(round(p * (len(s) - 1)))))
        return s[idx]

    return (
        f"        per-batch GiB/s  min {s[0]:6.3f}  p10 {pct(0.10):6.3f}  "
        f"p50 {pct(0.50):6.3f}  p90 {pct(0.90):6.3f}  max {s[-1]:6.3f}"
    )


def _format_segment(label: str, wall: float, proc: float, nbytes: int,
                    n_batches: int) -> str:
    """Format a 'warmup / steady state / overall' summary line.

    ``label`` is expected to include the batch count, e.g.
    "warmup (first 2 batches):".  Returns a "(no data)" placeholder if the
    segment is empty.
    """
    if n_batches == 0 or wall <= 0:
        return f"    {label:<36s}(no data)"
    bw = nbytes / wall / 1024**3
    cpu_pct = proc / wall * 100
    # cpu/(GiB/s) = (proc/wall*100) / (bytes/wall/1024**3)
    #             = proc * 100 * 1024**3 / bytes
    cpu_per_bw = cpu_pct / bw if bw > 0 else 0.0
    return (
        f"    {label:<36s}{bw:.4f} GiB/s, cpu {cpu_pct:.2f} %, "
        f"{cpu_per_bw:.4f} cpu%/(GiB/s)"
    )


# ---------------------------------------------------------------------------
# Benchmark core
# ---------------------------------------------------------------------------


def run_bench(
    urls: list[str],
    *,
    gpu_device: int,
    gpu: bool,
    align_buffer: bool,
    batch_size: int,
    warmup_batches: int,
    target_duration: float,
    json_out: bool = False,
    quiet: bool = False,
) -> None:

    assert batch_size >= 1, f"batch_size must be >= 1, got {batch_size}"
    assert warmup_batches >= 0, (
        f"warmup_batches must be >= 0, got {warmup_batches}"
    )

    # ---- set CUDA device --------------------------------------------------
    if gpu:
        import cupy as cp

        cp.cuda.Device(gpu_device).use()

    # ---- resolve files (untimed) ------------------------------------------
    files = _resolve_remote_files(urls)
    max_file_size = max(sz for _, sz in files)

    # Never allocate more buffers than there are files.
    n_buffers = min(batch_size, len(files))

    # ---- destination buffers (one per slot, sized to largest file) --------
    dsts: list = []
    _keepalives: list = []
    for _ in range(n_buffers):
        buf, ka = _allocate_buffer(
            max_file_size, gpu=gpu, aligned=align_buffer)
        dsts.append(buf)
        _keepalives.append(ka)

    # ---- print run header -------------------------------------------------
    total_file_bytes = sum(sz for _, sz in files)
    n_batches_per_pass = (len(files) + batch_size - 1) // batch_size
    print("--> benchmark config")
    print(f"    files           : {len(files)}")
    print(f"    total file size : {total_file_bytes / 1024**3:.4f} GiB")
    print(f"    max file size   : {max_file_size / 1024**3:.4f} GiB")
    if gpu:
        print(f"    gpu             : device {gpu_device}")
    else:
        print("    gpu             : False")
    print(f"    align_buffer    : {align_buffer}")
    print(f"    batch_size      : {batch_size}  ({n_buffers} buffer(s))")
    print(f"    warmup_batches  : {warmup_batches}")
    print(f"    target_duration : {target_duration} s")
    if len(files) <= 16:
        for url, sz in files:
            print(f"      {url}  ({sz / 1024**3:.4f} GiB)")
    print()

    # ---- benchmark loop ---------------------------------------------------
    import kvikio

    # Segmented accumulators -- warmup vs steady state.  Overall is the sum
    # of the two; computed at the end rather than tracked separately.
    warmup_wall = 0.0
    warmup_proc = 0.0
    warmup_bytes = 0
    warmup_count = 0

    steady_wall = 0.0
    steady_proc = 0.0
    steady_bytes = 0
    steady_count = 0

    warmup_bws: list[float] = []
    steady_bws: list[float] = []

    overall_open_time = 0.0
    overall_open_count = 0
    global_batch_idx = 0  # counts batches across pass boundaries

    pass_idx = 0
    done = False
    while not done:
        pass_idx += 1
        pass_read_time = 0.0
        pass_proc_time = 0.0
        pass_bytes = 0

        for batch_idx in range(n_batches_per_pass):
            start = batch_idx * batch_size
            end = min(start + batch_size, len(files))
            batch = files[start:end]

            # ---- untimed: open all files in the batch ---------------------
            handles = []
            for url, _sz in batch:
                t_open0 = time.perf_counter()
                fh = kvikio.RemoteFile.open_s3_url(url)
                t_open1 = time.perf_counter()
                handles.append(fh)
                overall_open_time += t_open1 - t_open0
                overall_open_count += 1

            # ---- timed: submit all preads, then await all -----------------
            # Wall outermost, process_time innermost: guarantees the wall
            # interval contains the proc interval so the ratio stays
            # physically bounded.
            t0_wall = time.perf_counter()
            t0_proc = time.process_time()
            futures = []
            for (_url, sz), fh, buf in zip(batch, handles, dsts):
                futures.append(fh.pread(buf, size=sz))
            for fut in futures:
                fut.get()
            t1_proc = time.process_time()
            t1_wall = time.perf_counter()

            # ---- untimed: close -------------------------------------------
            for fh in handles:
                fh.close()

            batch_time = t1_wall - t0_wall
            batch_proc = t1_proc - t0_proc
            batch_bytes = sum(sz for _, sz in batch)
            batch_bw = batch_bytes / batch_time / 1024**3
            batch_cpu_pct = (batch_proc / batch_time *
                             100) if batch_time > 0 else 0.0

            # Classify this batch into warmup or steady state.
            is_warmup = global_batch_idx < warmup_batches
            global_batch_idx += 1
            if is_warmup:
                warmup_bws.append(batch_bw)
                warmup_wall += batch_time
                warmup_proc += batch_proc
                warmup_bytes += batch_bytes
                warmup_count += 1
            else:
                steady_bws.append(batch_bw)
                steady_wall += batch_time
                steady_proc += batch_proc
                steady_bytes += batch_bytes
                steady_count += 1

            pass_read_time += batch_time
            pass_proc_time += batch_proc
            pass_bytes += batch_bytes

            tag = " [warmup]" if is_warmup else ""
            if not quiet:
                print(
                f"    [pass {pass_idx} batch {batch_idx + 1}/{n_batches_per_pass}]"
                f"{tag}  "
                f"{len(batch)} files, {batch_bytes / 1024**3:.4f} GiB in "
                f"{batch_time:.6f} s   {batch_bw:.4f} GiB/s   "
                f"cpu {batch_cpu_pct:.2f} %"
                )

            if (warmup_wall + steady_wall) >= target_duration:
                done = True
                break

        if pass_read_time > 0:
            pass_bw = pass_bytes / pass_read_time / 1024**3
            pass_cpu_pct = pass_proc_time / pass_read_time * 100
        else:
            pass_bw = 0.0
            pass_cpu_pct = 0.0
        print(
            f"--> pass {pass_idx} aggregate  "
            f"({pass_bytes / 1024**3:.4f} GiB in {pass_read_time:.4f} s):  "
            f"{pass_bw:.4f} GiB/s   cpu {pass_cpu_pct:.2f} %"
        )
        print()

    # ---- summary ----------------------------------------------------------
    overall_wall = warmup_wall + steady_wall
    overall_proc = warmup_proc + steady_proc
    overall_bytes = warmup_bytes + steady_bytes
    overall_count = warmup_count + steady_count

    print(
        f"--> overall summary  ({overall_bytes / 1024**3:.4f} GiB in "
        f"{overall_wall:.4f} s over {pass_idx} pass(es), "
        f"{overall_count} batch(es))"
    )
    print(_format_segment(
        f"warmup (first {warmup_count} batches):",
        warmup_wall, warmup_proc, warmup_bytes, warmup_count,
    ))
    print(_format_segment(
        f"steady state ({steady_count} batches):",
        steady_wall, steady_proc, steady_bytes, steady_count,
    ))
    if steady_bws:
        print(_percentiles(steady_bws))
    print(_format_segment(
        f"overall ({overall_count} batches):",
        overall_wall, overall_proc, overall_bytes, overall_count,
    ))
    if json_out:
        import json as _json
        print("JSON " + _json.dumps({
            "steady_bws": steady_bws,
            "steady_agg_bw": (steady_bytes / steady_wall / 1024**3)
            if steady_wall > 0 else 0.0,
            "steady_cpu_pct": (steady_proc / steady_wall * 100)
            if steady_wall > 0 else 0.0,
            "open_ms": (overall_open_time / overall_open_count * 1000)
            if overall_open_count else 0.0,
        }))

    if overall_open_count > 0:
        avg_open_ms = (overall_open_time / overall_open_count) * 1000
        print(
            f"    {'average open cost:':<36s}{avg_open_ms:.4f} ms  "
            f"({overall_open_count} opens, untimed)"
        )


# ---------------------------------------------------------------------------
# File list
# ---------------------------------------------------------------------------

# TPC-H sf3k lineitem: 180 parts of ~3.14 GiB each.
TPCH_TEMPLATE = "s3://rapids-presto-gpu/tpch/sf3k_v2_float/lineitem/part.{k}.parquet"
TPCH_NUM_PARTS = 180

# Single large object, useful for isolating per-request behaviour from
# per-object behaviour.
SINGLE_FILE = ["s3://kvikio-remote-io-dev/misc/8GiB.bin"]


def get_filenames(dataset: str, num_files: int) -> list[str]:
    """Build the URL list for the requested dataset.

    ``tpch`` cycles through distinct objects, which keeps each batch on a
    different set of S3 keys.  ``single`` reuses one object, so every batch
    hits the same key.
    """
    if dataset == "single":
        return SINGLE_FILE
    assert dataset == "tpch", f"unknown dataset {dataset!r}"
    n = min(num_files, TPCH_NUM_PARTS)
    return [TPCH_TEMPLATE.format(k=k) for k in range(n)]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="KvikIO S3 remote read microbenchmark")
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
        help="Read into a device buffer (default: --gpu)",
    )
    parser.add_argument(
        "--align-buffer",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Page-align the destination buffer (default: --align-buffer)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Number of files submitted concurrently per batch (default: 1)",
    )
    parser.add_argument(
        "--warmup-batches",
        type=int,
        default=1,
        help="Number of leading batches classified as warmup; the rest are "
        "steady state (default: 1)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=30.0,
        help="Target cumulative read duration in seconds (default: 30)",
    )
    parser.add_argument(
        "--dataset",
        choices=("tpch", "single"),
        default="tpch",
        help="tpch: 180 distinct ~3.14 GiB lineitem parts (default). "
        "single: one 8 GiB object, reused every batch.",
    )
    parser.add_argument(
        "--num-files",
        type=int,
        default=180,
        help="Number of tpch parts to use (default: 180)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress the per-batch lines, print only the summary",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit a machine-readable 'JSON {...}' summary line",
    )
    args = parser.parse_args()

    run_bench(
        get_filenames(args.dataset, args.num_files),
        gpu_device=args.gpu_device,
        gpu=args.gpu,
        align_buffer=args.align_buffer,
        batch_size=args.batch_size,
        warmup_batches=args.warmup_batches,
        target_duration=args.duration,
        json_out=args.json,
        quiet=args.quiet,
    )
