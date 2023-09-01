# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.

import argparse
import contextlib
import os
import os.path
import pathlib
import shutil
import statistics
import subprocess
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

if not kvikio.zarr.supported:
    raise RuntimeError(f"requires Zarr >={kvikio.zarr.MINIMUM_ZARR_VERSION}")

compressors = {
    "none": (None, None),
    "lz4": (numcodecs.blosc.Blosc(cname="lz4"), kvikio.zarr.LZ4()),
}


def drop_vm_cache(args):
    if args.drop_vm_cache:
        subprocess.check_output(["sudo /sbin/sysctl vm.drop_caches=3"], shell=True)


def create_src_data(args):
    return cupy.random.random(args.nelem, dtype=args.dtype)


def run_kvikio(args):
    dir_path = args.dir / "kvikio"
    shutil.rmtree(str(dir_path), ignore_errors=True)

    # Get the GPU compressor
    compressor = compressors[args.compressor][1]

    src = create_src_data(args)

    # Write
    drop_vm_cache(args)
    t0 = clock()
    z = zarr.create(
        shape=(args.nelem,),
        dtype=args.dtype,
        chunks=False,
        compressor=compressor,
        store=kvikio.zarr.GDSStore(dir_path),
        meta_array=cupy.empty(()),
    )
    z[:] = src
    os.sync()
    write_time = clock() - t0

    # Read
    drop_vm_cache(args)
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
    drop_vm_cache(args)
    t0 = clock()
    z = zarr.create(
        shape=(args.nelem,),
        dtype=args.dtype,
        chunks=False,
        compressor=compressor,
        store=zarr.DirectoryStore(dir_path),
        meta_array=numpy.empty(()),
    )
    z[:] = src.get()
    os.sync()
    write_time = clock() - t0

    # Read
    drop_vm_cache(args)
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

    kvikio.defaults.num_threads_reset(args.nthreads)
    props = kvikio.DriverProperties()
    try:
        import pynvml.smi

        nvsmi = pynvml.smi.nvidia_smi.getInstance()
    except ImportError:
        gpu_name = "Unknown (install pynvml)"
        mem_total = gpu_name
        bar1_total = gpu_name
    else:
        info = nvsmi.DeviceQuery()["gpu"][0]
        gpu_name = f"{info['product_name']} (dev #0)"
        mem_total = format_bytes(
            parse_bytes(
                str(info["fb_memory_usage"]["total"]) + info["fb_memory_usage"]["unit"]
            )
        )
        bar1_total = format_bytes(
            parse_bytes(
                str(info["bar1_memory_usage"]["total"])
                + info["bar1_memory_usage"]["unit"]
            )
        )
    gds_version = "N/A (Compatibility Mode)"
    if props.is_gds_available:
        gds_version = f"v{props.major_version}.{props.minor_version}"
    gds_config_json_path = os.path.realpath(
        os.getenv("CUFILE_ENV_PATH_JSON", "/etc/cufile.json")
    )
    drop_vm_cache_msg = str(args.drop_vm_cache)
    if not args.drop_vm_cache:
        drop_vm_cache_msg += " (use --drop-vm-cache for better accuracy!)"

    print("Roundtrip benchmark")
    print("----------------------------------")
    if kvikio.defaults.compat_mode():
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("   WARNING - KvikIO compat mode   ")
        print("      libcufile.so not used       ")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    elif not props.is_gds_available:
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("   WARNING - cuFile compat mode   ")
        print("         GDS not enabled          ")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print(f"GPU               | {gpu_name}")
    print(f"GPU Memory Total  | {mem_total}")
    print(f"BAR1 Memory Total | {bar1_total}")
    print(f"GDS driver        | {gds_version}")
    print(f"GDS config.json   | {gds_config_json_path}")
    print("----------------------------------")
    print(f"nbytes            | {args.nbytes} bytes ({format_bytes(args.nbytes)})")
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

    # Compute nelem
    args.nelem = args.nbytes // args.dtype.itemsize
    assert args.nbytes % args.dtype.itemsize == 0

    # Create a temporary directory if user didn't specify a directory
    temp_dir: Union[tempfile.TemporaryDirectory, ContextManager]
    if args.dir is None:
        temp_dir = tempfile.TemporaryDirectory()
        args.dir = pathlib.Path(temp_dir.name)
    else:
        temp_dir = contextlib.nullcontext()

    with temp_dir:
        main(args)
