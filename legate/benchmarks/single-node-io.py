# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.

import argparse
import contextlib
import os
import os.path
import pathlib
import statistics
import tempfile
from time import perf_counter as clock
from typing import ContextManager, Union

import cunumeric as num
from dask.utils import format_bytes, parse_bytes

import kvikio
import kvikio.defaults
import legate.core
from legate_kvikio import CuFile

runtime = legate.core.get_legate_runtime()


def run_cufile(args):
    """Single file and array"""
    file_path = args.dir / "kvikio-single-file"

    src = num.arange(args.nbytes, dtype="uint8")
    dst = num.empty_like(src)
    runtime.issue_execution_fence(block=True)

    # Write
    f = CuFile(file_path, flags="w")
    t0 = clock()
    f.write(src)
    f.close()
    runtime.issue_execution_fence(block=True)
    write_time = clock() - t0

    # Read
    f = CuFile(file_path, flags="r")
    t0 = clock()
    f.read(dst)
    f.close()
    runtime.issue_execution_fence(block=True)
    read_time = clock() - t0
    assert (src == dst).all()

    return read_time, write_time


API = {
    "cufile": run_cufile,
}


def main(args):

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
    print(f"directory         | {args.dir}")
    print(f"nthreads          | {args.nthreads}")
    print(f"nruns             | {args.nruns}")
    print(f"#CPUs             | {runtime.num_cpus}")
    print(f"#GPUs             | {runtime.num_gpus}")
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
        default=("cufile",),
        nargs="+",
        choices=tuple(API.keys()) + ("all",),
        help="List of APIs to use {%(choices)s}",
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
