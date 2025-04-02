# Copyright (c) 2021-2025, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.

import argparse
import contextlib
import pathlib
import shutil
import statistics
import tempfile
from time import perf_counter as clock
from typing import Any, ContextManager, Dict, Union

import cupy
from dask.utils import format_bytes, parse_bytes

import kvikio
import kvikio.buffer
import kvikio.defaults
from kvikio.benchmarks.utils import parse_directory, pprint_sys_info


def get_zarr_compressors() -> Dict[str, Any]:
    """Returns a dict of available Zarr compressors"""
    try:
        import kvikio.zarr
    except ImportError:
        return {}
    try:
        compressors = kvikio.zarr.nvcomp_compressors
    except AttributeError:
        # zarr-python 3.x
        return {}
    else:
        return {c.__name__.lower(): c for c in compressors}


def create_data(nbytes):
    """Return a random uint8 cupy array"""
    return cupy.arange(nbytes, dtype="uint8")


def run_cufile(args):
    """Single file and array"""

    file_path = args.dir / "kvikio-single-file"
    data = create_data(args.nbytes)
    if args.pre_register_buffer:
        kvikio.buffer.memory_register(data)

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
        kvikio.buffer.memory_deregister(data)

    return read_time, write_time


def run_cufile_multiple_files_multiple_arrays(args):
    """One file and array per thread"""

    chunksize = args.nbytes // args.nthreads
    assert args.nbytes % args.nthreads == 0, "--nbytes must be divisible by --nthreads"

    # Create a file path and CuPy array per thread
    file_path = str(args.dir / "cufile-p-%03d")
    arrays = [create_data(chunksize) for _ in range(args.nthreads)]
    if args.pre_register_buffer:
        for array in arrays:
            kvikio.buffer.memory_register(array)

    # Write
    files = [kvikio.CuFile(file_path % i, flags="w") for i in range(args.nthreads)]
    t0 = clock()
    futures = [f.pwrite(a, task_size=a.nbytes) for f, a in zip(files, arrays)]
    res = sum(f.get() for f in futures)
    del files
    write_time = clock() - t0
    assert res == args.nbytes

    # Read
    files = [kvikio.CuFile(file_path % i, flags="r") for i in range(args.nthreads)]
    t0 = clock()
    futures = [f.pread(a, task_size=a.nbytes) for f, a in zip(files, arrays)]
    res = sum(f.get() for f in futures)
    del files
    read_time = clock() - t0
    assert res == args.nbytes

    if args.pre_register_buffer:
        for array in arrays:
            kvikio.buffer.memory_deregister(array)

    return read_time, write_time


def run_cufile_multiple_files(args):
    """Single array but one file per thread"""

    chunksize = args.nbytes // args.nthreads
    assert args.nbytes % args.nthreads == 0, "--nbytes must be divisible by --nthreads"
    file_path = str(args.dir / "cufile-p-%03d")
    data = create_data(args.nbytes)
    if args.pre_register_buffer:
        kvikio.buffer.memory_register(data)

    # Write
    files = [kvikio.CuFile(file_path % i, flags="w") for i in range(args.nthreads)]
    t0 = clock()
    futures = [
        f.pwrite(data[i * chunksize : (i + 1) * chunksize]) for i, f in enumerate(files)
    ]
    res = sum(f.get() for f in futures)
    del files
    write_time = clock() - t0
    assert res == args.nbytes, f"IO mismatch, expected {args.nbytes} got {res}"

    # Read
    files = [kvikio.CuFile(file_path % i, flags="r") for i in range(args.nthreads)]
    t0 = clock()
    futures = [
        f.pread(data[i * chunksize : (i + 1) * chunksize]) for i, f in enumerate(files)
    ]
    res = sum(f.get() for f in futures)
    del files
    read_time = clock() - t0
    assert res == args.nbytes, f"IO mismatch, expected {args.nbytes} got {res}"

    if args.pre_register_buffer:
        kvikio.buffer.memory_deregister(data)

    return read_time, write_time


def run_cufile_multiple_arrays(args):
    """A single file but one array per thread"""

    chunksize = args.nbytes // args.nthreads
    assert args.nbytes % args.nthreads == 0, "--nbytes must be divisible by --nthreads"
    file_path = args.dir / "kvikio-multiple-arrays"

    # Create a CuPy array per thread
    arrays = [create_data(chunksize) for _ in range(args.nthreads)]
    if args.pre_register_buffer:
        for array in arrays:
            kvikio.buffer.memory_register(array)

    # Write
    f = kvikio.CuFile(file_path, flags="w")
    t0 = clock()
    futures = [
        f.pwrite(a, task_size=a.nbytes, file_offset=i * chunksize)
        for i, a in enumerate(arrays)
    ]
    res = sum(f.get() for f in futures)
    f.close()
    write_time = clock() - t0
    assert res == args.nbytes

    # Read
    f = kvikio.CuFile(file_path, flags="r")
    t0 = clock()
    futures = [f.pread(a, task_size=a.nbytes) for a in arrays]
    res = sum(f.get() for f in futures)
    f.close()
    read_time = clock() - t0
    assert res == args.nbytes

    if args.pre_register_buffer:
        for array in arrays:
            kvikio.buffer.memory_deregister(array)

    return read_time, write_time


def run_posix(args):
    """Use the posix API, no calls to kvikio"""

    file_path = args.dir / "posix-single-file"
    data = create_data(args.nbytes)

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


def run_zarr(args):
    """Use the Zarr API"""

    import zarr

    import kvikio.zarr

    dir_path = args.dir / "zarr"
    if not kvikio.zarr.supported:
        raise RuntimeError(f"requires Zarr >={kvikio.zarr.MINIMUM_ZARR_VERSION}")

    compressor = None
    if args.zarr_compressor is not None:
        compressor = get_zarr_compressors()[args.zarr_compressor]()

    a = create_data(args.nbytes)

    shutil.rmtree(str(dir_path), ignore_errors=True)

    # Write
    t0 = clock()
    z = zarr.array(
        a,
        chunks=False,
        compressor=compressor,
        store=kvikio.zarr.GDSStore(dir_path),
        meta_array=cupy.empty(()),
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
    "zarr": run_zarr,
    "posix": run_posix,
    "cufile-mfma": run_cufile_multiple_files_multiple_arrays,
    "cufile-mf": run_cufile_multiple_files,
    "cufile-ma": run_cufile_multiple_arrays,
}


def main(args):
    cupy.cuda.set_allocator(None)  # Disable CuPy's default memory pool
    cupy.arange(10)  # Make sure CUDA is initialized

    kvikio.defaults.set("num_threads", args.nthreads)

    print("Roundtrip benchmark")
    print("----------------------------------")
    pprint_sys_info()
    print("----------------------------------")
    print(f"nbytes            | {args.nbytes} bytes ({format_bytes(args.nbytes)})")
    print(f"4K aligned        | {args.nbytes % 4096 == 0}")
    print(f"pre-reg-buf       | {args.pre_register_buffer}")
    print(f"directory         | {args.dir}")
    print(f"nthreads          | {args.nthreads}")
    print(f"nruns             | {args.nruns}")
    if args.zarr_compressor is not None:
        print(f"Zarr compressor   | {args.zarr_compressor}")
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
        "--pre-register-buffer",
        action="store_true",
        default=False,
        help="Enable pre-register of device buffer",
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
    parser.add_argument(
        "--zarr-compressor",
        metavar="COMPRESSOR",
        default=None,
        choices=tuple(get_zarr_compressors().keys()),
        help=(
            "Set a nvCOMP compressor to use with Zarr "
            "{%(choices)s} (default: %(default)s)"
        ),
    )

    args = parser.parse_args()
    if "all" in args.api:
        args.api = tuple(API.keys())

    # Create a temporary directory if user didn't specify a directory
    temp_dir: Union[tempfile.TemporaryDirectory, ContextManager]
    if args.dir is None:
        temp_dir = tempfile.TemporaryDirectory()
        args.dir = pathlib.Path(temp_dir.name)
    else:
        temp_dir = contextlib.nullcontext()

    with temp_dir:
        main(args)
