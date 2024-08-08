# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.

import argparse
import contextlib
import multiprocessing
import os
import socket
import statistics
import sys
import time
from functools import partial
from typing import ContextManager
from urllib.parse import urlparse

import boto3
import cupy
import numpy
from dask.utils import format_bytes

import kvikio
import kvikio.defaults


def get_local_port() -> int:
    """Return an available port"""
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def start_s3_server(server_address, lifetime=3600):
    from moto.server import ThreadedMotoServer

    # Silence the activity info from ThreadedMotoServer
    sys.stderr = open("/dev/null", "w")
    url = urlparse(server_address)
    server = ThreadedMotoServer(ip_address=url.hostname, port=url.port)
    server.start()
    time.sleep(lifetime)


@contextlib.contextmanager
def local_s3_server(server_address):
    # Use fake aws credentials
    os.environ["AWS_ACCESS_KEY_ID"] = "foobar_key"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "foobar_secret"
    os.environ["AWS_SECURITY_TOKEN"] = "foobar_security_token"
    os.environ["AWS_SESSION_TOKEN"] = "foobar_session_token"
    os.environ["AWS_DEFAULT_REGION"] = "us-east-1"
    os.environ["AWS_ENDPOINT_URL"] = server_address
    p = multiprocessing.Process(target=start_s3_server, args=(server_address,))
    p.start()
    yield
    p.kill()


def run_numpy_like(args, xp):
    # Upload data to S3 server
    data = numpy.arange(args.nelem, dtype=args.dtype)
    recv = xp.empty_like(data)

    client = boto3.client("s3", endpoint_url=args.server_address)
    client.create_bucket(Bucket=args.bucket, ACL="public-read-write")

    client.put_object(Bucket=args.bucket, Key="data1", Body=bytes(data))

    def run() -> float:
        t0 = time.perf_counter()
        with kvikio.RemoteFile(bucket_name=args.bucket, object_name="data1") as f:
            res = f.read(recv)
        t1 = time.perf_counter()
        assert res == args.nbytes, f"IO mismatch, expected {args.nbytes} got {res}"
        xp.testing.assert_array_equal(data, recv)
        return t1 - t0

    for _ in range(args.nruns):
        yield run()


def run_cudf(args, use_kvikio_s3):
    import cudf

    # Upload data to S3 server
    data = cupy.random.rand(args.nelem).astype(args.dtype)
    df = cudf.DataFrame({"a": data})
    df.to_parquet(f"s3://{args.bucket}/data1")

    def run() -> float:
        t0 = time.perf_counter()
        cudf.read_parquet(f"s3://{args.bucket}/data1", use_kvikio_s3=use_kvikio_s3)
        t1 = time.perf_counter()
        return t1 - t0

    for _ in range(args.nruns):
        yield run()


API = {
    "cupy-kvikio": partial(run_numpy_like, xp=cupy),
    "numpy-kvikio": partial(run_numpy_like, xp=numpy),
    "cudf-kvikio": partial(run_cudf, use_kvikio_s3=True),
    "cudf-fsspec": partial(run_cudf, use_kvikio_s3=False),
}


def main(args):
    cupy.cuda.set_allocator(None)  # Disable CuPy's default memory pool
    cupy.arange(10)  # Make sure CUDA is initialized

    kvikio.defaults.num_threads_reset(args.nthreads)
    print("Roundtrip benchmark")
    print("-------------------------------------")
    print(f"nelem       | {args.nelem} ({format_bytes(args.nbytes)})")
    print(f"dtype       | {args.dtype}")
    print(f"nthreads    | {args.nthreads}")
    print(f"nruns       | {args.nruns}")
    print(f"server      | {args.server_address}")
    print("=====================================")

    # Run each benchmark using the requested APIs
    for api in args.api:
        res = []
        for elapsed in API[api](args):
            res.append(elapsed)

        def pprint_api_res(name, samples):
            samples = [args.nbytes / s for s in samples]  # Convert to throughput
            mean = statistics.mean(samples) if len(samples) > 1 else samples[0]
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
    parser = argparse.ArgumentParser(description="Roundtrip benchmark")
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
        "--server-address",
        metavar="ADDRESS",
        default="LOCAL",
        type=str,
        help=(
            "Address of the S3 server e.g. http://127.0.0.1:4200. "
            "By default, a local server is launched and used."
        ),
    )
    parser.add_argument(
        "--bucket",
        metavar="NAME",
        default="kvikio-s3-benchmark",
        type=str,
        help="Name of the AWS S3 bucket to use (default: %(default)s).",
    )
    parser.add_argument(
        "--api",
        metavar="API",
        default=("cupy",),
        nargs="+",
        choices=tuple(API.keys()) + ("all",),
        help="List of APIs to use {%(choices)s}",
    )
    args = parser.parse_args()
    args.nbytes = args.nelem * args.dtype.itemsize
    if "all" in args.api:
        args.api = tuple(API.keys())

    assert args.server_address == "LOCAL"  # TODO: support non-local servers

    ctx: ContextManager = contextlib.nullcontext()
    if args.server_address == "LOCAL":
        args.server_address = f"http://127.0.0.1:{get_local_port()}"
        ctx = local_s3_server(args.server_address)
    with ctx:
        main(args)
