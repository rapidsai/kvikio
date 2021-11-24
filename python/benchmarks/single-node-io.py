# Copyright (c) 2021-2022, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.

import argparse
import os
import pathlib
import tempfile
from concurrent.futures import ThreadPoolExecutor
from time import perf_counter as clock
from typing import Optional

import cupy
from dask.utils import format_bytes, parse_bytes

import cufile


def run_posix(args):
    data = cupy.arange(args.nbytes // 8, dtype="int64")

    # Write
    f = open(args.file_path, "wb")
    t0 = clock()
    f.write(data.tobytes())
    f.close()
    write_time = clock() - t0

    # Read
    f = open(args.file_path, "rb")
    t0 = clock()
    cupy.fromfile(f, dtype="int64", count=len(data))
    f.close()
    read_time = clock() - t0

    return read_time, write_time


def run_cufile_single_thread(args):
    data = cupy.arange(args.nbytes // 8, dtype="int64")
    if args.pre_register_buffer:
        cufile.memory_register(data)

    # Write
    f = cufile.CuFile(file_path=args.file_path, flags="w")
    t0 = clock()
    f.write(data)
    f.close()
    write_time = clock() - t0

    # Read
    f = cufile.CuFile(file_path=args.file_path, flags="r")
    t0 = clock()
    f.read(data)
    f.close()
    read_time = clock() - t0

    if args.pre_register_buffer:
        cufile.memory_deregister(data)

    return read_time, write_time


def run_cufile_parallel(args):
    current_device = cupy.cuda.Device().id
    executor = ThreadPoolExecutor(
        max_workers=args.nthreads,
        initializer=lambda: cupy.cuda.Device(current_device).use(),
    )

    data = cupy.arange(args.nbytes // 8, dtype="int64")
    if args.pre_register_buffer:
        cufile.memory_register(data)

    chunks = []
    file_offsets = []
    for i in range(args.nthreads):
        chunk_size = len(data) // args.nthreads
        offset = i * chunk_size
        chunks.append(data[offset : offset + chunk_size])
        file_offsets.append(offset * 8)

    def write_parallel(rank):
        f.write(chunks[rank], file_offset=file_offsets[rank])

    def read_parallel(rank):
        f.read(chunks[rank], file_offset=file_offsets[rank])

    # Write
    f = cufile.CuFile(file_path=args.file_path, flags="w")
    t0 = clock()
    list(executor.map(write_parallel, range(args.nthreads)))
    f.close()
    write_time = clock() - t0

    # Read
    f = cufile.CuFile(file_path=args.file_path, flags="r")
    t0 = clock()
    list(executor.map(read_parallel, range(args.nthreads)))
    f.close()
    read_time = clock() - t0

    if args.pre_register_buffer:
        cufile.memory_deregister(data)

    return read_time, write_time


def main(args):
    res = {}
    if args.nthreads == 1:
        run_cufile = run_cufile_single_thread
    else:
        run_cufile = run_cufile_parallel

    try:
        if not args.no_posix_run:
            read, write = run_posix(args)
            res["posix"] = (args.nbytes / read, args.nbytes / write)
        read, write = run_cufile(args)
        res["cufile"] = (args.nbytes / read, args.nbytes / write)
    finally:
        try:
            os.remove(args.file_path)
        except FileNotFoundError:
            pass

    props = cufile.DriverProperties()
    nvml = cufile.NVML()
    mem_total, _ = nvml.get_memory()
    bar1_total, _ = nvml.get_bar1_memory()
    gds_version = "N/A (Compatibility Mode)"
    if props.is_gds_availabe:
        gds_version = f"v{props.major_version}.{props.minor_version}"

    print("Roundtrip benchmark")
    print("----------------------------------")
    if not props.is_gds_availabe:
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("             WARNING              ")
        print("   Compat mode, GDS not enabled   ")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

    print(f"GPU               | {nvml.get_name()}")
    print(f"GPU Memory Total  | {format_bytes(mem_total)}")
    print(f"BAR1 Memory Total | {format_bytes(bar1_total)}")
    print(f"GDS driver        | {gds_version}")
    print("----------------------------------")
    print(f"nbytes            | {args.nbytes} bytes ({format_bytes(args.nbytes)})")
    print(f"4K aligned        | {args.nbytes % 4096 == 0}")
    print(f"pre-reg-buf       | {args.pre_register_buffer}")
    print(f"file-path         | {args.file_path}")
    print(f"nthreads          | {args.nthreads}")
    print("==================================")
    print(f"CuFile Read       | {format_bytes(res['cufile'][0])}/s")
    print(f"CuFile Write      | {format_bytes(res['cufile'][1])}/s")
    if "posix" in res:
        print(f"Posix Read        | {format_bytes(res['posix'][0])}/s")
        print(f"Posix Write       | {format_bytes(res['posix'][1])}/s")


if __name__ == "__main__":

    def default_file_path(path: Optional[str] = None) -> pathlib.Path:
        if path is None:
            f = tempfile.NamedTemporaryFile(delete=False)
            path = f.name
            f.close()
        return path

    parser = argparse.ArgumentParser(description="Roundtrip benchmark")
    parser.add_argument(
        "-n",
        "--nbytes",
        metavar="BYTES",
        default="10 MiB",
        type=parse_bytes,
        help="Message size, which must be a multiple of 8 (default 10 Mb).",
    )
    parser.add_argument(
        "-f",
        "--file-path",
        metavar="PATH",
        default=default_file_path(),
        type=pathlib.Path,
        help="Path to r/w file (default tempfile.NamedTemporaryFile)",
    )
    parser.add_argument(
        "--no-pre-register-buffer",
        action="store_true",
        default=False,
        help="Disable pre-register of device buffer",
    )
    parser.add_argument(
        "--no-posix-run", action="store_true", default=False, help="Disable POSIX run",
    )
    parser.add_argument(
        "-t",
        "--nthreads",
        metavar="THREADS",
        default=1,
        type=int,
        help="Number of threads to use (default 1).",
    )
    args = parser.parse_args()
    args.pre_register_buffer = args.no_pre_register_buffer is False
    main(args)
