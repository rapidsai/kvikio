# Copyright (c) 2023-2025, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.

import argparse
import contextlib
import os
import os.path
import pathlib
import shutil
import statistics
import tempfile
from time import perf_counter as clock
from typing import ContextManager, Union

import cupy
import numcodecs.blosc
import numpy
import zarr
from dask.utils import format_bytes, parse_bytes

import kvikio
import kvikio.defaults
import kvikio.zarr
from kvikio.benchmarks.utils import drop_vm_cache, parse_directory, pprint_sys_info

if not kvikio.zarr.supported:
    raise RuntimeError(f"requires Zarr >={kvikio.zarr.MINIMUM_ZARR_VERSION}")

compressors = {
    "none": (None, None),
    "lz4": (numcodecs.blosc.Blosc(cname="lz4"), kvikio.zarr.LZ4()),
}


def create_src_data(args):
    return cupy.random.random(args.nelem, dtype=args.dtype)


def run_kvikio(args):
    dir_path = args.dir / "kvikio"
    shutil.rmtree(str(dir_path), ignore_errors=True)

    # Get the GPU compressor
    compressor = compressors[args.compressor][1]

    src = create_src_data(args)

    # Write
    if args.drop_vm_cache:
        drop_vm_cache()
    t0 = clock()
    z = zarr.create(
        shape=(args.nelem,),
        chunks=(args.chunksize,),
        dtype=args.dtype,
        compressor=compressor,
        store=kvikio.zarr.GDSStore(dir_path),
        meta_array=cupy.empty(()),
    )
    z[:] = src
    os.sync()
    write_time = clock() - t0

    # Read
    if args.drop_vm_cache:
        drop_vm_cache()
    t0 = clock()
    res = z[:]
    read_time = clock() - t0
    assert res.nbytes == args.nbytes

    return read_time, write_time


def run_posix(args):
    dir_path = args.dir / "posix"
    shutil.rmtree(str(dir_path), ignore_errors=True)

    # Get the CPU compressor
    compressor = compressors[args.compressor][0]

    src = create_src_data(args)

    # Write
    if args.drop_vm_cache:
        drop_vm_cache()
    t0 = clock()
    z = zarr.create(
        shape=(args.nelem,),
        chunks=(args.chunksize,),
        dtype=args.dtype,
        compressor=compressor,
        store=zarr.DirectoryStore(dir_path),
        meta_array=numpy.empty(()),
    )
    z[:] = src.get()
    os.sync()
    write_time = clock() - t0

    # Read
    if args.drop_vm_cache:
        drop_vm_cache()
    t0 = clock()
    res = cupy.asarray(z[:])
    read_time = clock() - t0
    assert res.nbytes == args.nbytes

    return read_time, write_time


API = {
    "kvikio": run_kvikio,
    "posix": run_posix,
}


def main(args):
    cupy.cuda.set_allocator(None)  # Disable CuPy's default memory pool
    cupy.arange(10)  # Make sure CUDA is initialized

    kvikio.defaults.set("num_threads", args.nthreads)
    drop_vm_cache_msg = str(args.drop_vm_cache)
    if not args.drop_vm_cache:
        drop_vm_cache_msg += " (use --drop-vm-cache for better accuracy!)"
    chunksize = args.chunksize * args.dtype.itemsize

    print("Zarr-IO Benchmark")
    print("----------------------------------")
    pprint_sys_info()
    print("----------------------------------")
    print(f"nbytes            | {args.nbytes} bytes ({format_bytes(args.nbytes)})")
    print(f"chunksize         | {chunksize} bytes ({format_bytes(chunksize)})")
    print(f"4K aligned        | {args.nbytes % 4096 == 0}")
    print(f"drop-vm-cache     | {drop_vm_cache_msg}")
    print(f"directory         | {args.dir}")
    print(f"nthreads          | {args.nthreads}")
    print(f"nruns             | {args.nruns}")
    print(f"compressor        | {args.compressor}")
    print("==================================")

    # Run each benchmark using the requested APIs
    for api in args.api:
        rs = []
        ws = []
        for _ in range(args.n_warmup_runs):
            read, write = API[api](args)
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
        "--chunksize",
        metavar="BYTES",
        default="10 MiB",
        type=parse_bytes,
        help="Chunk size (default: %(default)s).",
    )
    parser.add_argument(
        "--dtype",
        default="float32",
        type=numpy.dtype,
        help="NumPy datatype to use (default: '%(default)s')",
    )
    parser.add_argument(
        "-d",
        "--dir",
        metavar="PATH",
        default=None,
        type=parse_directory,
        help="Path to the directory to r/w from (default: tempfile.TemporaryDirectory)",
    )
    parser.add_argument(
        "--nruns",
        metavar="RUNS",
        default=1,
        type=int,
        help="Number of runs per API (default: %(default)s).",
    )
    parser.add_argument(
        "--n-warmup-runs",
        default=0,
        type=int,
        help="Number of warmup runs (default: %(default)s).",
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
        default=("kvikio", "posix"),
        nargs="+",
        choices=tuple(API.keys()) + ("all",),
        help="List of APIs to use {%(choices)s}",
    )
    parser.add_argument(
        "--compressor",
        metavar="COMPRESSOR",
        default="none",
        choices=tuple(compressors.keys()),
        help=(
            "Set a nvCOMP compressor to use with Zarr "
            "{%(choices)s} (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--drop-vm-cache",
        action="store_true",
        default=False,
        help=(
            "Drop the VM cache between writes and reads, "
            "requires sudo access to /sbin/sysctl"
        ),
    )

    args = parser.parse_args()
    if "all" in args.api:
        args.api = tuple(API.keys())

    # Check if size is divisible by size of datatype
    assert args.nbytes % args.dtype.itemsize == 0
    assert args.chunksize % args.dtype.itemsize == 0

    # Compute/convert to number of elements
    args.nelem = args.nbytes // args.dtype.itemsize
    args.chunksize = args.chunksize // args.dtype.itemsize

    # Create a temporary directory if user didn't specify a directory
    temp_dir: Union[tempfile.TemporaryDirectory, ContextManager]
    if args.dir is None:
        temp_dir = tempfile.TemporaryDirectory()
        args.dir = pathlib.Path(temp_dir.name)
    else:
        temp_dir = contextlib.nullcontext()

    with temp_dir:
        main(args)
