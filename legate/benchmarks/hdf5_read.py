# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.

import argparse
import contextlib
import operator
import pathlib
import tempfile
from time import perf_counter as clock
from typing import ContextManager

import h5py
import numpy as np

DATASET = "dataset-hdf5-read"


def try_open_hdf5_array(filepath, shape, chunks, dtype):
    try:
        with h5py.File(filepath, "r") as f:
            a = f[DATASET]
        chunks = chunks or a.chunks
        if a.shape == shape and a.chunks == chunks and a.dtype == dtype:
            return a
    except FileNotFoundError:
        pass
    return None


def create_hdf5_array(filepath, shape, chunks, dtype=np.float64) -> None:
    ret = try_open_hdf5_array(filepath, shape, chunks, dtype)
    if ret is None:
        # Write array using h5py
        with h5py.File(filepath, "w") as f:
            f.create_dataset(DATASET, chunks=chunks, data=np.random.random(shape))
    print(f"HDF5 '{filepath}': shape: {shape}, " f"chunks: {chunks}, dtype: {dtype}")


@contextlib.contextmanager
def dask_h5py(args):
    import h5py
    from dask import array as da
    from dask_cuda import LocalCUDACluster
    from distributed import Client

    def f():
        t0 = clock()
        with h5py.File(args.dir / "A", "r") as af:
            with h5py.File(args.dir / "B", "r") as bf:
                a = da.from_array(af[DATASET], chunks=af[DATASET].chunks)
                b = da.from_array(bf[DATASET], chunks=bf[DATASET].chunks)
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
    from legate_kvikio.kerchunk import hdf5_read

    def f():
        get_legate_runtime().issue_execution_fence(block=True)
        t0 = clock()
        a = hdf5_read(args.dir / "A", dataset_name=DATASET)
        b = hdf5_read(args.dir / "B", dataset_name=DATASET)
        c = args.op(num, a, b)
        int(c.sum())
        t1 = clock()
        return t1 - t0

    yield f


API = {
    "dask-h5py": dask_h5py,
    "legate": run_legate,
}


OP = {"add": operator.add, "matmul": operator.matmul}


def main(args):
    create_hdf5_array(args.dir / "A", chunks=(args.c, args.c), shape=(args.m, args.m))
    create_hdf5_array(args.dir / "B", chunks=(args.c, args.c), shape=(args.m, args.m))

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
        "-c",
        default=None,
        type=int,
        help="Dimension of the squired chunk (CxC) (default: M/10).",
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
    if args.c is None:
        args.c = args.m // 10

    # Create a temporary directory if user didn't specify a directory
    temp_dir: tempfile.TemporaryDirectory | ContextManager
    if args.dir is None:
        temp_dir = tempfile.TemporaryDirectory()
        args.dir = pathlib.Path(temp_dir.name)
    else:
        temp_dir = contextlib.nullcontext()

    with temp_dir:
        main(args)
