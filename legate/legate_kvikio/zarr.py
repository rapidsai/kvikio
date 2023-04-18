# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.

from __future__ import annotations

import math
import pathlib
from typing import Optional, Tuple

import cunumeric
import zarr.core

from .tile import read_tiles, write_tiles


def _get_padded_array(zarr_ary: zarr.Array) -> Optional[cunumeric.ndarray]:
    if all(s % c == 0 for s, c in zip(zarr_ary.shape, zarr_ary.chunks)):
        return None  # Already aligned

    padded_shape = tuple(
        math.ceil(s / c) * c for s, c in zip(zarr_ary.shape, zarr_ary.chunks)
    )
    return cunumeric.empty(shape=padded_shape, dtype=zarr_ary.dtype)


def write_array(
    ary: cunumeric.ndarray,
    dirpath: pathlib.Path | str,
    chunks: Optional[int | Tuple[int]],
    compressor=None,
) -> None:
    dirpath = pathlib.Path(dirpath)
    if compressor is not None:
        raise NotImplementedError("compressor isn't supported")

    # We use Zarr to write the meta data
    zarr_ary = zarr.open_array(
        dirpath,
        shape=ary.shape,
        dtype=ary.dtype,
        mode="w",
        chunks=chunks,
        compressor=compressor,
    )
    padded_ary = _get_padded_array(zarr_ary)
    if padded_ary is None:
        write_tiles(ary, dirpath=dirpath, tile_shape=zarr_ary.chunks)
    else:
        padded_ary[...] = -1  # TODO: remove line
        padded_ary[tuple(slice(s) for s in zarr_ary.shape)] = ary
        write_tiles(padded_ary, dirpath=dirpath, tile_shape=zarr_ary.chunks)


def read_array(dirpath: pathlib.Path | str) -> cunumeric.ndarray:
    dirpath = pathlib.Path(dirpath)

    # We use Zarr to read the meta data
    zarr_ary = zarr.open_array(dirpath, mode="r")
    if zarr_ary.compressor is not None:
        raise NotImplementedError("compressor isn't supported")

    padded_ary = _get_padded_array(zarr_ary)
    if padded_ary is None:
        ret = cunumeric.empty(shape=zarr_ary.shape, dtype=zarr_ary.dtype)
        ret[...] = -1  # TODO: remove line
        read_tiles(ret, dirpath=dirpath, tile_shape=zarr_ary.chunks)
    else:
        read_tiles(padded_ary, dirpath=dirpath, tile_shape=zarr_ary.chunks)
        ret = padded_ary[tuple(slice(s) for s in zarr_ary.shape)]
    return ret
