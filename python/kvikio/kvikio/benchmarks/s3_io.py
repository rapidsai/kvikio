# Copyright (c) 2024-2025, NVIDIA CORPORATION. All rights reserved.
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


def start_s3_server(lifetime: int):
    """Start a server and run it for `lifetime` minutes.
    NB: to stop before `lifetime`, kill the process/thread running this function.
    """
    from moto.server import ThreadedMotoServer

    # Silence the activity info from ThreadedMotoServer
    sys.stderr = open(os.devnull, "w")
    url = urlparse(os.environ["AWS_ENDPOINT_URL"])
    server = ThreadedMotoServer(ip_address=url.hostname, port=url.port)
    server.start()
    time.sleep(lifetime)


@contextlib.contextmanager
def local_s3_server(lifetime: int):
    """Start a server and run it for `lifetime` minutes or kill it on context exit"""
    # Use fake aws credentials
    os.environ["AWS_ACCESS_KEY_ID"] = "foobar_key"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "foobar_secret"
    os.environ["AWS_DEFAULT_REGION"] = "us-east-1"
    p = multiprocessing.Process(target=start_s3_server, args=(lifetime,))
    p.start()
    yield
    p.kill()


def create_client_and_bucket():
    client = boto3.client("s3", endpoint_url=os.getenv("AWS_ENDPOINT_URL", None))
    try:
        bucket_names = {bucket["Name"] for bucket in client.list_buckets()["Buckets"]}
        if args.bucket not in bucket_names:
            client.create_bucket(Bucket=args.bucket, ACL="public-read-write")
    except Exception:
        print(
            "Problem accessing the S3 server? using wrong credentials? Try setting "
            "AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, and/or AWS_ENDPOINT_URL. Also, "
            "if the bucket doesn't exist, make sure you have the required permission. "
            "Alternatively, use the bundled server `--use-bundled-server`:\n",
            file=sys.stderr,
            flush=True,
        )
        raise
    return client


def run_numpy_like(args, xp):
    # Upload data to S3 server
    data = numpy.arange(args.nelem, dtype=args.dtype)
    recv = xp.empty_like(data)

    client = create_client_and_bucket()
    client.put_object(Bucket=args.bucket, Key="data", Body=bytes(data))
    url = f"s3://{args.bucket}/data"

    def run() -> float:
        t0 = time.perf_counter()
        with kvikio.RemoteFile.open_s3_url(url) as f:
            res = f.read(recv)
        t1 = time.perf_counter()
        assert res == args.nbytes, f"IO mismatch, expected {args.nbytes} got {res}"
        xp.testing.assert_array_equal(data, recv)
        return t1 - t0

    for _ in range(args.nruns):
        yield run()


def run_cudf(args, kvikio_remote_io: bool):
    import cudf

    cudf.set_option("kvikio_remote_io", kvikio_remote_io)
    url = f"s3://{args.bucket}/data"

    # Upload data to S3 server
    create_client_and_bucket()
    data = cupy.random.rand(args.nelem).astype(args.dtype)
    df = cudf.DataFrame({"a": data})
    df.to_parquet(url)

    def run() -> float:
        t0 = time.perf_counter()
        cudf.read_parquet(url)
        t1 = time.perf_counter()
        return t1 - t0

    for _ in range(args.nruns):
        yield run()


API = {
    "cupy": partial(run_numpy_like, xp=cupy),
    "numpy": partial(run_numpy_like, xp=numpy),
    "cudf-kvikio": partial(run_cudf, kvikio_remote_io=True),
    "cudf-fsspec": partial(run_cudf, kvikio_remote_io=False),
}


def main(args):
    cupy.cuda.set_allocator(None)  # Disable CuPy's default memory pool
    cupy.arange(10)  # Make sure CUDA is initialized

    os.environ["KVIKIO_NTHREADS"] = str(args.nthreads)
    kvikio.defaults.set("num_threads", args.nthreads)

    print("Remote S3 benchmark")
    print("--------------------------------------")
    print(f"nelem       | {args.nelem} ({format_bytes(args.nbytes)})")
    print(f"dtype       | {args.dtype}")
    print(f"nthreads    | {args.nthreads}")
    print(f"nruns       | {args.nruns}")
    print(f"file        | s3://{args.bucket}/data")
    if args.use_bundled_server:
        print("--------------------------------------")
        print("Using the bundled local server is slow")
        print("and can be misleading. Consider using")
        print("a local MinIO or official S3 server.")
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
        "--use-bundled-server",
        action="store_true",
        help="Launch and use a local slow S3 server (ThreadedMotoServer).",
    )
    parser.add_argument(
        "--bundled-server-lifetime",
        metavar="SECONDS",
        default=3600,
        type=int,
        help="Maximum lifetime of the bundled server (default: %(default)s).",
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
        default="all",
        nargs="+",
        choices=tuple(API.keys()) + ("all",),
        help="List of APIs to use {%(choices)s} (default: %(default)s).",
    )
    args = parser.parse_args()
    args.nbytes = args.nelem * args.dtype.itemsize
    if "all" in args.api:
        args.api = tuple(API.keys())

    ctx: ContextManager = contextlib.nullcontext()
    if args.use_bundled_server:
        os.environ["AWS_ENDPOINT_URL"] = f"http://127.0.0.1:{get_local_port()}"
        ctx = local_s3_server(args.bundled_server_lifetime)
    with ctx:
        main(args)
