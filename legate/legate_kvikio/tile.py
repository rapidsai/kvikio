# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.

from __future__ import annotations

import pathlib
from typing import Optional, Tuple

import cunumeric

import legate.core.types as types
from legate.core import Rect

from .library_description import TaskOpCode, context
from .utils import get_legate_store


def write_tiles(
    ary: cunumeric.ndarray,
    dirpath: pathlib.Path | str,
    tile_shape: Tuple[int],
    tile_start: Optional[Tuple[int]] = None,
) -> None:
    dirpath = pathlib.Path(dirpath)
    if tile_start is None:
        tile_start = (0,) * len(tile_shape)

    if any(d % c != 0 for d, c in zip(ary.shape, tile_shape)):
        raise ValueError(
            f"The tile shape {tile_shape} must be "
            f"divisible with the array shape {ary.shape}"
        )

    # Partition the array into even tiles
    store_partition = get_legate_store(ary).partition_by_tiling(tile_shape)

    # Use the partition's color shape as the launch shape so there will be
    # one task for each tile
    launch_shape = store_partition.partition.color_shape
    task = context.create_manual_task(
        TaskOpCode.TILE_WRITE,
        launch_domain=Rect(launch_shape),
    )
    task.add_input(store_partition)
    task.add_scalar_arg(str(dirpath), types.string)
    task.add_scalar_arg(tile_shape, (types.uint64,))
    task.add_scalar_arg(tile_start, (types.uint64,))
    task.execute()


def read_tiles(
    ary: cunumeric.ndarray,
    dirpath: pathlib.Path | str,
    tile_shape: Tuple[int],
    tile_start: Optional[Tuple[int]] = None,
) -> None:
    dirpath = pathlib.Path(dirpath)
    if tile_start is None:
        tile_start = (0,) * len(tile_shape)

    if any(d % c != 0 for d, c in zip(ary.shape, tile_shape)):
        raise ValueError(
            f"The tile shape {tile_shape} must be "
            f"divisible with the array shape {ary.shape}"
        )

    # Partition the array into even tiles
    store_partition = get_legate_store(ary).partition_by_tiling(tile_shape)

    # Use the partition's color shape as the launch shape so there will be
    # one task for each tile
    launch_shape = store_partition.partition.color_shape
    task = context.create_manual_task(
        TaskOpCode.TILE_READ,
        launch_domain=Rect(launch_shape),
    )
    task.add_output(store_partition)
    task.add_scalar_arg(str(dirpath), types.string)
    task.add_scalar_arg(tile_shape, (types.uint64,))
    task.add_scalar_arg(tile_start, (types.uint64,))
    task.execute()
