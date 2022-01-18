# Copyright (c) 2021-2022, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.

import argparse
import contextlib
import functools
import os
import pathlib
import shutil
import tempfile
from time import perf_counter as clock

import cupy
from dask.utils import format_bytes, parse_bytes

import cufile


def run_cufile(args):
    file_path = args.dir / "cufile-single-file"
    cufile.set_num_threads(args.nthreads)
    data = cupy.arange(args.nbytes, dtype="uint8")
    if args.pre_register_buffer:
        cufile.memory_register(data)

    # Write
    f = cufile.CuFile(file_path=file_path, flags="w")
    t0 = clock()
    res = f.write(data)
    f.close()
    write_time = clock() - t0
    assert res == args.nbytes, f"IO mismatch, expected {args.nbytes} got {res}"

    # Read
    f = cufile.CuFile(file_path=file_path, flags="r")
    t0 = clock()
    res = f.read(data)
    f.close()
    read_time = clock() - t0
    assert res == args.nbytes, f"IO mismatch, expected {args.nbytes} got {res}"

    if args.pre_register_buffer:
        cufile.memory_deregister(data)

    return read_time, write_time


def run_cufile_multiple_files(args):
    """Read/write to a file per thread"""

    chunksize = args.nbytes // args.nthreads
    assert args.nbytes % args.nthreads == 0, "--nbytes must be divisible by --nthreads"

    # Create a file path and CuPy array per thread
    file_path = str(args.dir / "cufile-p-%03d")
    arrays = [cupy.arange(chunksize, dtype="uint8") for _ in range(args.nthreads)]
    if args.pre_register_buffer:
        for array in arrays:
            cufile.memory_register(array)

    # Write
    files = [
        cufile.CuFile(file_path=file_path % i, flags="w") for i in range(args.nthreads)
    ]
    t0 = clock()
    futures = [f.pwrite(a, nthreads=1) for f, a in zip(files, arrays)]
    res = sum(f.get() for f in futures)
    write_time = clock() - t0
    assert res == args.nbytes

    # Read
    files = [
        cufile.CuFile(file_path=file_path % i, flags="r") for i in range(args.nthreads)
    ]
    t0 = clock()
    futures = [f.pread(a, nthreads=1) for f, a in zip(files, arrays)]
    res = sum(f.get() for f in futures)
    read_time = clock() - t0
    assert res == args.nbytes

    if args.pre_register_buffer:
        for array in arrays:
            cufile.memory_deregister(array)

    return read_time, write_time


def run_posix(args):
    file_path = args.dir / "posix-single-file"
    data = cupy.arange(args.nbytes, dtype="uint8")

    # Write
    f = open(file_path, "wb")
    t0 = clock()
    res = f.write(data.tobytes())
    f.close()
    write_time = clock() - t0
    assert res == args.nbytes

    # Read
    f = open(file_path, "rb")
    t0 = clock()
    a = cupy.fromfile(f, dtype="uint8", count=len(data))
    f.close()
    read_time = clock() - t0
    assert a.nbytes == args.nbytes
    assert res == args.nbytes, f"IO mismatch, expected {args.nbytes} got {a.nbytes}"

    return read_time, write_time


def run_zarr(store_type, args):
    import zarr
    import zarr.cupy

    import cufile.zarr

    cufile.set_num_threads(args.nthreads)
    a = cupy.arange(args.nbytes // 8, dtype="int64")

    # Retrieve the store and compressor to use based on `store_type`
    shutil.rmtree(str(args.dir / store_type), ignore_errors=True)
    store, compressor = {
        "gds": (cufile.zarr.GDSStore(args.dir / store_type), None),
        "posix": (
            zarr.DirectoryStore(args.dir / store_type),
            zarr.cupy.CuPyCPUCompressor(),
        ),
    }[store_type]

    # Write
    t0 = clock()
    z = zarr.array(
        a, chunks=False, compressor=compressor, store=store, meta_array=cupy.empty(())
    )
    write_time = clock() - t0

    # Read
    t0 = clock()
    z[:]
    read_time = clock() - t0

    return read_time, write_time


API = {
    "cufile": run_cufile,
    "zarr-gds": functools.partial(run_zarr, "gds"),
    "zarr-posix": functools.partial(run_zarr, "posix"),
    "posix": run_posix,
    "cufile-p": run_cufile_multiple_files,
}


def main(args):
    cupy.cuda.set_allocator(None)  # Disable CuPy's default memory pool
    cufile.set_num_threads(args.nthreads)
    results = {}
    for api in args.api:
        read, write = API[api](args)
        results[api] = (args.nbytes / read, args.nbytes / write)
    props = cufile.DriverProperties()
    nvml = cufile.NVML()
    mem_total, _ = nvml.get_memory()
    bar1_total, _ = nvml.get_bar1_memory()
    gds_version = "N/A (Compatibility Mode)"
    if props.is_gds_availabe:
        gds_version = f"v{props.major_version}.{props.minor_version}"
    gds_config_json_path = os.getenv("CUFILE_ENV_PATH_JSON", "/etc/cufile.json")

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
    print(f"GDS config.json   | {gds_config_json_path}")
    print("----------------------------------")
    print(f"nbytes            | {args.nbytes} bytes ({format_bytes(args.nbytes)})")
    print(f"4K aligned        | {args.nbytes % 4096 == 0}")
    print(f"pre-reg-buf       | {args.pre_register_buffer}")
    print(f"diretory          | {args.dir}")
    print(f"nthreads          | {args.nthreads}")
    print("==================================")
    for api, (r, w) in results.items():
        print(f"{api} read".ljust(18) + f"| {format_bytes(r)}/s")
        print(f"{api} write".ljust(18) + f"| {format_bytes(w)}/s")


if __name__ == "__main__":

    def parse_directory(x):
        if x is None:
            return x
        else:
            p = pathlib.Path(x)
            if not p.is_dir():
                raise argparse.ArgumentTypeError("Must be a directory")
            return p

    parser = argparse.ArgumentParser(description="Roundtrip benchmark")
    parser.add_argument(
        "-n",
        "--nbytes",
        metavar="BYTES",
        default="10 MiB",
        type=parse_bytes,
        help="Message size, which must be a multiple of 8 (default: %(default)s).",
    )
    parser.add_argument(
        "-d",
        "--dir",
        metavar="PATH",
        default=None,
        type=parse_directory,
        help="Path to the diretory to r/w from (default: tempfile.TemporaryDirectory)",
    )
    parser.add_argument(
        "--no-pre-register-buffer",
        action="store_true",
        default=False,
        help="Disable pre-register of device buffer",
    )
    parser.add_argument(
        "-t",
        "--nthreads",
        metavar="THREADS",
        default=1,
        type=int,
        help="Number of threads to use (default: %(default)s).",
    )
    parser.add_argument(
        "--api",
        default=("cufile", "posix"),
        nargs="+",
        choices=tuple(API.keys()),
        help="List of APIs to use",
    )

    args = parser.parse_args()
    args.pre_register_buffer = args.no_pre_register_buffer is False

    # Create a temporary directory if user didn't specify a directory
    if args.dir is None:
        temp_dir = tempfile.TemporaryDirectory()
        args.dir = pathlib.Path(temp_dir.name)
    else:
        temp_dir = contextlib.nullcontext()

    with temp_dir:
        main(args)
