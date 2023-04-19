# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.

import argparse
import contextlib
import pathlib
import tempfile
from time import perf_counter as clock
from typing import ContextManager

import numpy as np
import zarr
from zarr.errors import ArrayNotFoundError


def try_open_zarr_array(dirpath, shape, chunks, dtype):
    try:
        a = zarr.open_array(dirpath, mode="r")
        chunks = chunks or a.chunks
        if a.shape == shape and a.chunks == chunks and a.dtype == dtype:
            return a
    except ArrayNotFoundError:
        pass
    return None


def create_zarr_array(dirpath, shape, chunks=None, dtype=np.float64) -> None:
    ret = try_open_zarr_array(dirpath, shape, chunks, dtype)
    if ret is None:
        ret = zarr.open_array(
            dirpath,
            shape=shape,
            dtype=dtype,
            mode="w",
            chunks=chunks,
            compressor=None,
        )
        ret[:] = np.random.random(shape)

    print(
        f"Zarr '{ret.store.path}': shape: {ret.shape}, "
        f"chunks: {ret.chunks}, dtype: {ret.dtype}"
    )


@contextlib.contextmanager
def run_dask(args):
    from dask import array as da
    from dask_cuda import LocalCUDACluster
    from distributed import Client

    def f():
        t0 = clock()
        a = da.from_zarr(args.dir / "A")
        b = da.from_zarr(args.dir / "B")
        c = da.matmul(a, b)
        int(c.sum().compute())
        t1 = clock()
        return t1 - t0

    with LocalCUDACluster(n_workers=args.n_workers) as cluster:
        with Client(cluster):
            yield f


@contextlib.contextmanager
def run_legate(args):
    import cunumeric as num
    from legate_kvikio.zarr import read_array

    from legate.core import get_legate_runtime

    def f():
        get_legate_runtime().issue_execution_fence(block=True)
        t0 = clock()
        a = read_array(args.dir / "A")
        b = read_array(args.dir / "B")
        c = num.matmul(a, b)
        int(c.sum())
        t1 = clock()
        return t1 - t0

    yield f


API = {
    "dask": run_dask,
    "legate": run_legate,
}


def main(args):
    create_zarr_array(args.dir / "A", shape=(args.m, args.k))
    create_zarr_array(args.dir / "B", shape=(args.k, args.n))

    with API[args.api](args) as f:
        for _ in range(args.nruns):
            elapsed = f()
            print("elapsed: ", elapsed)


if __name__ == "__main__":

    def parse_directory(x):
        if x is None:
            return x
        else:
            p = pathlib.Path(x)
            if not p.is_dir():
                raise argparse.ArgumentTypeError("Must be a directory")
            return p

    parser = argparse.ArgumentParser(
        description="Matrix multiplication of two Zarr files"
    )
    parser.add_argument(
        "-m",
        default=100,
        type=int,
        help="Size of the M dimension (default: %(default)s).",
    )
    parser.add_argument(
        "-n",
        default=0,
        type=int,
        help="Size of the N dimension. If not set, using the value of `-m`.",
    )
    parser.add_argument(
        "-k",
        default=0,
        type=int,
        help="Size of the K dimension. If not set, using the value of `-m`.",
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
        help="Number of runs (default: %(default)s).",
    )
    parser.add_argument(
        "--api",
        metavar="API",
        default="dask",
        choices=tuple(API.keys()),
        help="API to use {%(choices)s}",
    )
    parser.add_argument(
        "--n-workers",
        default=1,
        type=int,
        help="Number of workers (default: %(default)s).",
    )
    args = parser.parse_args()

    # Default `-n` and `-k` to the value of `-m`
    args.n = args.n or args.m
    args.k = args.k or args.m

    # Create a temporary directory if user didn't specify a directory
    temp_dir: tempfile.TemporaryDirectory | ContextManager
    if args.dir is None:
        temp_dir = tempfile.TemporaryDirectory()
        args.dir = pathlib.Path(temp_dir.name)
    else:
        temp_dir = contextlib.nullcontext()

    with temp_dir:
        main(args)
