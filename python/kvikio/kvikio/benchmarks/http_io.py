# Copyright (c) 2024-2025, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.

import argparse
import contextlib
import pathlib
import statistics
import tempfile
import time
from functools import partial

import cupy
import numpy
from dask.utils import format_bytes

import kvikio
import kvikio.defaults
from kvikio.utils import LocalHttpServer


def run_numpy_like(args, xp):
    src = numpy.arange(args.nelem, dtype=args.dtype)
    src.tofile(args.server_root_path / "data")
    dst = xp.empty_like(src)
    url = f"{args.server_url}/data"

    def run() -> float:
        t0 = time.perf_counter()
        with kvikio.RemoteFile.open_http(url, nbytes=src.nbytes) as f:
            res = f.read(dst)
        t1 = time.perf_counter()
        assert res == args.nbytes, f"IO mismatch, expected {args.nbytes} got {res}"
        xp.testing.assert_array_equal(src, dst)
        return t1 - t0

    for _ in range(args.nruns):
        yield run()


API = {
    "cupy": partial(run_numpy_like, xp=cupy),
    "numpy": partial(run_numpy_like, xp=numpy),
}


def main(args):
    cupy.cuda.set_allocator(None)  # Disable CuPy's default memory pool
    cupy.arange(10)  # Make sure CUDA is initialized

    kvikio.defaults.set("num_threads", args.nthreads)
    print("Roundtrip benchmark")
    print("--------------------------------------")
    print(f"nelem       | {args.nelem} ({format_bytes(args.nbytes)})")
    print(f"dtype       | {args.dtype}")
    print(f"nthreads    | {args.nthreads}")
    print(f"nruns       | {args.nruns}")
    print(f"server      | {args.server}")
    if args.server is None:
        print("--------------------------------------")
        print("WARNING: the bundled server is slow, ")
        print("consider using --server.")
    print("======================================")

    # Run each benchmark using the requested APIs
    for api in args.api:
        res = []
        for elapsed in API[api](args):
            res.append(elapsed)

        def pprint_api_res(name, samples):
            # Convert to throughput
            samples = [args.nbytes / s for s in samples]
            mean = statistics.harmonic_mean(samples) if len(samples) > 1 else samples[0]
            ret = f"{api}-{name}".ljust(18)
            ret += f"| {format_bytes(mean).rjust(10)}/s".ljust(14)
            if len(samples) > 1:
                stdev = statistics.stdev(samples) / mean * 100
                ret += " ± %5.2f %%" % stdev
                ret += " ("
                for sample in samples:
                    ret += f"{format_bytes(sample)}/s, "
                ret = ret[:-2] + ")"  # Replace trailing comma
            return ret

        print(pprint_api_res("read", res))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HTTP benchmark")
    parser.add_argument(
        "-n",
        "--nelem",
        metavar="NELEM",
        default="1024",
        type=int,
        help="Number of elements (default: %(default)s).",
    )
    parser.add_argument(
        "--dtype",
        metavar="DATATYPE",
        default="float32",
        type=numpy.dtype,
        help="The data type of each element (default: %(default)s).",
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
        "--server",
        default=None,
        help=(
            "Connect to an external http server as opposed "
            "to the bundled (very slow) HTTP server. "
            "Remember to also set --server-root-path."
        ),
    )
    parser.add_argument(
        "--server-root-path",
        default=None,
        help="Path to the root directory that `--server` serves (local path).",
    )
    parser.add_argument(
        "--bundled-server-lifetime",
        metavar="SECONDS",
        default=3600,
        type=int,
        help="Maximum lifetime of the bundled server (default: %(default)s).",
    )
    parser.add_argument(
        "--api",
        metavar="API",
        default=list(API.keys())[0],  # defaults to the first API
        nargs="+",
        choices=tuple(API.keys()) + ("all",),
        help="List of APIs to use {%(choices)s} (default: %(default)s).",
    )
    args = parser.parse_args()
    args.nbytes = args.nelem * args.dtype.itemsize
    if "all" in args.api:
        args.api = tuple(API.keys())

    with contextlib.ExitStack() as context_stack:
        if args.server is None:
            # Create a tmp dir for the bundled server to serve
            temp_dir = tempfile.TemporaryDirectory()
            args.bundled_server_root_dir = pathlib.Path(temp_dir.name)
            context_stack.enter_context(temp_dir)

            # Create the bundled server
            bundled_server = LocalHttpServer(
                root_path=args.bundled_server_root_dir,
                range_support=True,
                max_lifetime=args.bundled_server_lifetime,
            )
            context_stack.enter_context(bundled_server)
            args.server_url = bundled_server.url
            args.server_root_path = args.bundled_server_root_dir
        else:
            args.server_url = args.server
            if args.server_root_path is None:
                raise ValueError("please set --server-root-path")
            args.server_root_path = pathlib.Path(args.server_root_path)
        main(args)
