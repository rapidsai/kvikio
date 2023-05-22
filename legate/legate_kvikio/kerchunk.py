# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.

from __future__ import annotations

import itertools
import math
import pathlib

import cunumeric
import fsspec
import zarr.core
import zarr.hierarchy
from kerchunk.hdf import SingleHdf5ToZarr

from .tile import read_tiles_by_offsets
from .zarr import get_padded_array


def hdf5_read(filepath: pathlib.Path | str, dataset_name: str) -> cunumeric.ndarray:
    """Read an HDF5 array from disk using KvikIO

    Notes
    -----
    The returned array is padded to make its shape divisible by the shape of
    the Zarr chunks on disk (if not already). This means that the returned
    Legate store can be larger than the returned cuNumeric array.

    Parameters
    ----------
    filepath
        File path to the hdf5 file.

    Return
    ------
        The cuNumeric array read from disk.
    """
    filepath = pathlib.Path(filepath)

    # TODO: look for already generated kerchunk annotations
    annotations = SingleHdf5ToZarr(filepath, inline_threshold=0).translate()

    # Load annotations
    zarr_group = zarr.open(fsspec.get_mapper("reference://", fo=annotations))
    zarr_ary: zarr.Array = zarr_group[dataset_name]
    if zarr_ary.compressor is not None:
        raise NotImplementedError("compressor isn't supported")

    # Extract offset and bytes for each chunk
    refs = annotations["refs"]
    offsets = []
    sizes = []
    for chunk_coord in itertools.product(
        *(range(math.ceil(s / c)) for s, c in zip(zarr_ary.shape, zarr_ary.chunks))
    ):
        key = zarr_ary._chunk_key(chunk_coord)
        _, offset, nbytes = refs[key]
        offsets.append(offset)
        sizes.append(nbytes)

    padded_ary = get_padded_array(zarr_ary)
    if padded_ary is None:
        ret = cunumeric.empty(shape=zarr_ary.shape, dtype=zarr_ary.dtype)
        read_tiles_by_offsets(
            ret,
            filepaths=[filepath],
            offsets=tuple(offsets),
            sizes=tuple(sizes),
            tile_shape=zarr_ary.chunks,
        )
    else:
        read_tiles_by_offsets(
            padded_ary,
            filepaths=[filepath],
            offsets=tuple(offsets),
            sizes=tuple(sizes),
            tile_shape=zarr_ary.chunks,
        )
        ret = padded_ary[tuple(slice(s) for s in zarr_ary.shape)]
    return ret
