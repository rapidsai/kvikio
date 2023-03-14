# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.

import struct
from enum import IntEnum
from typing import Any

import numpy as np

import legate.core.types as types
from legate.core import Store, get_legate_runtime

from .library import user_context, user_lib


class TaskOpCode(IntEnum):
    WRITE = user_lib.cffi.OP_WRITE
    READ = user_lib.cffi.OP_READ


def _get_legate_store(input: Any) -> Store:
    """Extracts a Legate store from any object
       implementing the legate data interface

    Args:
        input (Any): The input object

    Returns:
        Store: The extracted Legate store
    """
    if isinstance(input, Store):
        return input
    data = input.__legate_data_interface__["data"]
    field = next(iter(data))
    array = data[field]
    _, store = array.stores()
    return store


def zero() -> Store:
    """Create a Legates store representing a single zero scalar

    Returns:
        Store: A Legate store representing a scalar zero
    """
    data = bytearray(8)
    buf = struct.pack(f"{len(data)}s", data)
    future = get_legate_runtime().create_future(buf, len(buf))
    return user_context.create_store(
        types.int64,
        shape=(1,),
        storage=future,
        optimize_scalar=True,
    )


def get_written_nbytes(input: Store) -> int:
    """Blocking call to get the value of `write()`

    Args:
        input (Store): The Legate store encapsulating a scalar

    Returns:
        float: A Python scalar
    """
    buf = input.storage.get_buffer(np.int64().itemsize)
    result = np.frombuffer(buf, dtype=np.int64, count=1)
    return int(result[0])


def write(path: str, obj: Any) -> Store:
    """Write data to disk using KvikIO

    Args:
        obj (Any): A Legate store or any object implementing
                     the Legate data interface.
    """
    input = _get_legate_store(obj)
    task = user_context.create_auto_task(TaskOpCode.WRITE)
    task.add_scalar_arg(path, types.string)
    task.add_input(input)
    task.set_side_effect(True)
    # TODO: return a handle instead.
    output = zero()
    task.add_reduction(output, types.ReductionOp.ADD)
    task.execute()
    return output


def read(path: str, obj: Any) -> None:
    """Read data from disk using KvikIO

    Args:
        obj (Any): A Legate store or any object implementing
                     the Legate data interface.
    """
    output = _get_legate_store(obj)
    task = user_context.create_auto_task(TaskOpCode.READ)
    task.add_scalar_arg(path, types.string)
    task.add_output(output)
    task.set_side_effect(True)
    task.execute()
