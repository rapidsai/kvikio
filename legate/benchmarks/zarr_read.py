# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.

import argparse
import contextlib
import functools
import pathlib
import tempfile
from time import perf_counter as clock
from typing import ContextManager

import numpy as np
import zarr
from zarr.errors import ArrayNotFoundError

from kvikio.zarr import GDSStore


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
def run_dask(args, *, use_cupy):
    from dask import array as da
    from dask_cuda import LocalCUDACluster
    from distributed import Client

    def f():
        t0 = clock()
        if use_cupy:
            import cupy

            az = zarr.open_array(GDSStore(args.dir / "A"), meta_array=cupy.empty(()))
            bz = zarr.open_array(GDSStore(args.dir / "B"), meta_array=cupy.empty(()))
        else:
            az = args.dir / "A"
            bz = args.dir / "B"

        a = da.from_zarr(az)
        b = da.from_zarr(bz)
        c = args.op(da, a, b)

        int(c.sum().compute())
        t1 = clock()
        return t1 - t0

    with LocalCUDACluster(n_workers=args.n_workers) as cluster:
        with Client(cluster):
            yield f


@contextlib.contextmanager
def run_legate(args):
    import cunumeric as num

    from legate.core import get_legate_runtime
    from legate_kvikio.zarr import read_array

    def f():
        get_legate_runtime().issue_execution_fence(block=True)
        t0 = clock()
        a = read_array(args.dir / "A")
        b = read_array(args.dir / "B")
        c = args.op(num, a, b)
        int(c.sum())
        t1 = clock()
        return t1 - t0

    yield f


API = {
    "dask-cpu": functools.partial(run_dask, use_cupy=False),
    "dask-gpu": functools.partial(run_dask, use_cupy=True),
    "legate": run_legate,
}

OP = {"add": lambda xp, a, b: a + b, "matmul": lambda xp, a, b: xp.matmul(a, b)}


def main(args):
    create_zarr_array(args.dir / "A", shape=(args.m, args.m))
    create_zarr_array(args.dir / "B", shape=(args.m, args.m))

    timings = []
    with API[args.api](args) as f:
        for _ in range(args.n_warmup_runs):
            elapsed = f()
            print("elapsed[warmup]: ", elapsed)
        for i in range(args.nruns):
            elapsed = f()
            print(f"elapsed[run #{i}]: ", elapsed)
            timings.append(elapsed)
    print(f"elapsed mean: {np.mean(timings):.5}s (std: {np.std(timings):.5}s)")


if __name__ == "__main__":

    def parse_directory(x):
        if x is None:
            return x
        else:
            p = pathlib.Path(x)
            if not p.is_dir():
                raise argparse.ArgumentTypeError("Must be a directory")
            return p

    parser = argparse.ArgumentParser(description="Matrix operation on two Zarr files")
    parser.add_argument(
        "-m",
        default=100,
        type=int,
        help="Dimension of the two squired input matrix (MxM) (default: %(default)s).",
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
        default=tuple(API.keys())[0],
        choices=tuple(API.keys()),
        help="API to use {%(choices)s}",
    )
    parser.add_argument(
        "--n-workers",
        default=1,
        type=int,
        help="Number of workers (default: %(default)s).",
    )
    parser.add_argument(
        "--op",
        metavar="OP",
        default=tuple(OP.keys())[0],
        choices=tuple(OP.keys()),
        help="Operation to run {%(choices)s}",
    )
    parser.add_argument(
        "--n-warmup-runs",
        default=1,
        type=int,
        help="Number of warmup runs (default: %(default)s).",
    )

    args = parser.parse_args()
    args.op = OP[args.op]  # Parse the operation argument

    # Create a temporary directory if user didn't specify a directory
    temp_dir: tempfile.TemporaryDirectory | ContextManager
    if args.dir is None:
        temp_dir = tempfile.TemporaryDirectory()
        args.dir = pathlib.Path(temp_dir.name)
    else:
        temp_dir = contextlib.nullcontext()

    with temp_dir:
        main(args)
