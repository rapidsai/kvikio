# Copyright (c) 2021-2022, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.

import argparse
import contextlib
import functools
import os
import os.path
import pathlib
import shutil
import statistics
import tempfile
from time import perf_counter as clock
from typing import Union

import cupy
from dask.utils import format_bytes, parse_bytes

import kvikio
import kvikio.thread_pool


def run_cufile(args):
    """Single file and array"""

    file_path = args.dir / "kvikio-single-file"
    data = cupy.arange(args.nbytes, dtype="uint8")
    if args.pre_register_buffer:
        kvikio.memory_register(data)

    # Write
    f = kvikio.CuFile(file_path, flags="w")
    t0 = clock()
    res = f.write(data)
    f.close()
    write_time = clock() - t0
    assert res == args.nbytes, f"IO mismatch, expected {args.nbytes} got {res}"

    # Read
    f = kvikio.CuFile(file_path, flags="r")
    t0 = clock()
    res = f.read(data)
    f.close()
    read_time = clock() - t0
    assert res == args.nbytes, f"IO mismatch, expected {args.nbytes} got {res}"

    if args.pre_register_buffer:
        kvikio.memory_deregister(data)

    return read_time, write_time


def run_cufile_multiple_files_multiple_arrays(args):
    """One file and array per thread"""

    chunksize = args.nbytes // args.nthreads
    assert args.nbytes % args.nthreads == 0, "--nbytes must be divisible by --nthreads"

    # Create a file path and CuPy array per thread
    file_path = str(args.dir / "cufile-p-%03d")
    arrays = [cupy.arange(chunksize, dtype="uint8") for _ in range(args.nthreads)]
    if args.pre_register_buffer:
        for array in arrays:
            kvikio.memory_register(array)

    # Write
    files = [kvikio.CuFile(file_path % i, flags="w") for i in range(args.nthreads)]
    t0 = clock()
    futures = [f.pwrite(a, ntasks=1) for f, a in zip(files, arrays)]
    res = sum(f.get() for f in futures)
    write_time = clock() - t0
    assert res == args.nbytes

    # Read
    files = [kvikio.CuFile(file_path % i, flags="r") for i in range(args.nthreads)]
    t0 = clock()
    futures = [f.pread(a, ntasks=1) for f, a in zip(files, arrays)]
    res = sum(f.get() for f in futures)
    read_time = clock() - t0
    assert res == args.nbytes

    if args.pre_register_buffer:
        for array in arrays:
            kvikio.memory_deregister(array)

    return read_time, write_time


def run_cufile_multiple_files(args):
    """Single array but one file per thread"""

    chunksize = args.nbytes // args.nthreads
    assert args.nbytes % args.nthreads == 0, "--nbytes must be divisible by --nthreads"
    file_path = str(args.dir / "cufile-p-%03d")
    data = cupy.arange(args.nbytes, dtype="uint8")
    if args.pre_register_buffer:
        kvikio.memory_register(data)

    # Write
    files = [kvikio.CuFile(file_path % i, flags="w") for i in range(args.nthreads)]
    t0 = clock()
    futures = [
        f.pwrite(data[i * chunksize : (i + 1) * chunksize]) for i, f in enumerate(files)
    ]
    res = sum(f.get() for f in futures)
    write_time = clock() - t0
    assert res == args.nbytes, f"IO mismatch, expected {args.nbytes} got {res}"

    # Read
    files = [kvikio.CuFile(file_path % i, flags="r") for i in range(args.nthreads)]
    t0 = clock()
    futures = [
        f.pread(data[i * chunksize : (i + 1) * chunksize]) for i, f in enumerate(files)
    ]
    res = sum(f.get() for f in futures)
    read_time = clock() - t0
    assert res == args.nbytes, f"IO mismatch, expected {args.nbytes} got {res}"

    if args.pre_register_buffer:
        kvikio.memory_deregister(data)

    return read_time, write_time


def run_cufile_multiple_arrays(args):
    """A single file but one array per thread"""

    chunksize = args.nbytes // args.nthreads
    assert args.nbytes % args.nthreads == 0, "--nbytes must be divisible by --nthreads"
    file_path = args.dir / "kvikio-multiple-arrays"

    # Create a CuPy array per thread
    arrays = [cupy.arange(chunksize, dtype="uint8") for _ in range(args.nthreads)]
    if args.pre_register_buffer:
        for array in arrays:
            kvikio.memory_register(array)

    # Write
    f = kvikio.CuFile(file_path, flags="w")
    t0 = clock()
    futures = [
        f.pwrite(a, ntasks=1, file_offset=i * chunksize) for i, a in enumerate(arrays)
    ]
    res = sum(f.get() for f in futures)
    write_time = clock() - t0
    assert res == args.nbytes

    # Read
    f = kvikio.CuFile(file_path, flags="r")
    t0 = clock()
    futures = [f.pread(a, ntasks=1) for a in arrays]
    res = sum(f.get() for f in futures)
    read_time = clock() - t0
    assert res == args.nbytes

    if args.pre_register_buffer:
        for array in arrays:
            kvikio.memory_deregister(array)

    return read_time, write_time


def run_posix(args):
    """Use the posix API, no calls to kvikio"""

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
    """Use the Zarr API"""

    import zarr
    import zarr.cupy

    import kvikio.zarr

    a = cupy.arange(args.nbytes, dtype="uint8")

    # Retrieve the store and compressor to use based on `store_type`
    shutil.rmtree(str(args.dir / store_type), ignore_errors=True)
    store, compressor = {
        "gds": (kvikio.zarr.GDSStore(args.dir / store_type), None),
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
    res = z[:]
    read_time = clock() - t0
    assert res.nbytes == args.nbytes

    return read_time, write_time


API = {
    "cufile": run_cufile,
    "zarr-gds": functools.partial(run_zarr, "gds"),
    "zarr-posix": functools.partial(run_zarr, "posix"),
    "posix": run_posix,
    "cufile-mfma": run_cufile_multiple_files_multiple_arrays,
    "cufile-mf": run_cufile_multiple_files,
    "cufile-ma": run_cufile_multiple_arrays,
}


def main(args):
    cupy.cuda.set_allocator(None)  # Disable CuPy's default memory pool
    cupy.arange(10)  # Make sure CUDA is initialized

    kvikio.thread_pool.reset_num_threads(args.nthreads)
    props = kvikio.DriverProperties()
    nvml = kvikio.NVML()
    mem_total, _ = nvml.get_memory()
    bar1_total, _ = nvml.get_bar1_memory()
    gds_version = "N/A (Compatibility Mode)"
    if props.is_gds_availabe:
        gds_version = f"v{props.major_version}.{props.minor_version}"
    gds_config_json_path = os.path.realpath(
        os.getenv("CUFILE_ENV_PATH_JSON", "/etc/cufile.json")
    )

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
    print(f"nruns             | {args.nruns}")
    print("==================================")

    # Run each benchmark using the requested APIs
    for api in args.api:
        rs = []
        ws = []
        for _ in range(args.nruns):
            read, write = API[api](args)
            rs.append(args.nbytes / read)
            ws.append(args.nbytes / write)

        def pprint_api_res(name, samples):
            mean = statistics.mean(samples) if len(samples) > 1 else samples[0]
            ret = f"{api} {name}".ljust(18)
            ret += f"| {format_bytes(mean).rjust(10)}/s".ljust(14)
            if len(samples) > 1:
                stdev = statistics.stdev(samples) / mean * 100
                ret += " ± %5.2f %%" % stdev
                ret += " ("
                for sample in samples:
                    ret += f"{format_bytes(sample)}/s, "
                ret = ret[:-2] + ")"  # Replace trailing comma
            return ret

        print(pprint_api_res("read", rs))
        print(pprint_api_res("write", ws))


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
        "--nruns",
        metavar="RUNS",
        default=1,
        type=int,
        help="Number of runs per API (default: %(default)s).",
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
        metavar="API",
        default=("cufile", "posix"),
        nargs="+",
        choices=tuple(API.keys()) + ("all",),
        help="List of APIs to use {%(choices)s}",
    )

    args = parser.parse_args()
    args.pre_register_buffer = args.no_pre_register_buffer is False
    if "all" in args.api:
        args.api = tuple(API.keys())

    # Create a temporary directory if user didn't specify a directory
    temp_dir: Union[tempfile.TemporaryDirectory, contextlib.nullcontext]
    if args.dir is None:
        temp_dir = tempfile.TemporaryDirectory()
        args.dir = pathlib.Path(temp_dir.name)
    else:
        temp_dir = contextlib.nullcontext()

    with temp_dir:
        main(args)
