# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.

from __future__ import annotations

import pathlib
from typing import Any

import legate.core.types as types

from .library_description import TaskOpCode, context
from .utils import get_legate_store


class CuFile:
    """File handle for GPUDirect Storage (GDS)"""

    def __init__(self, file: pathlib.Path | str, flags: str = "r"):
        """Open file for GDS IO operations

        The file is opened on demand when calling `.read()` and `.write()`.

        Warning, Legate-KvikIO doesn't maintain a file descriptor thus the file path
        to the file must not change while opened by this handle.

        Parameters
        ----------
        file: pathlib.Path or str
            Path-like object giving the pathname (absolute or relative to the current
            working directory) of the file to be opened and registered.
        flags: str, optional
            "r" -> "open for reading (default)"
            "w" -> "open for writing, truncating the file first"
            "+" -> "open for updating (reading and writing)"
        """
        assert "a" not in flags
        self._closed = False
        self._filepath = str(file)
        self._flags = flags

        # We open the file here in order to:
        #   * trigger exceptions here instead of in the Legate tasks, which
        #     forces the Python interpreter to exit.
        #   * create or truncate files opened in "w" mode, which is required
        #     because `TaskOpCode.WRITE` always opens the file in "r+" mode.
        with open(self._filepath, mode=flags):
            pass

    def close(self) -> None:
        """Deregister the file and close the file"""
        self._closed = True

    @property
    def closed(self) -> bool:
        return self._closed

    def fileno(self) -> int:
        raise RuntimeError("Legate-KvikIO doesn't expose any file descriptor")

    def open_flags(self) -> int:
        """Get the flags of the file descriptor (see open(2))"""
        raise RuntimeError("Legate-KvikIO doesn't expose any file descriptor")

    def __enter__(self) -> CuFile:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def read(self, buf: Any) -> None:
        """Reads specified buffer from the file into device or host memory

        Warning, the size of `buf` must be greater than the size of the file.

        Parameters
        ----------
        buf: legate-store-like (1-dimensional)
            A Legate store or any object implementing `__legate_data_interface__` to
            read into.
        """
        assert not self._closed
        if "r" not in self._flags and "+" not in self._flags:
            raise ValueError(f"Cannot read a file opened with flags={self._flags}")

        output = get_legate_store(buf)
        task = context.create_auto_task(TaskOpCode.READ)
        task.add_scalar_arg(self._filepath, types.string)
        task.add_output(output)
        task.set_side_effect(True)
        task.execute()

    def write(self, buf: Any) -> None:
        """Writes specified buffer from device or host memory to the file

        Hint, if a subsequent operation read this file, insert a fence in between
        such as `legate.core.get_legate_runtime().issue_execution_fence(block=False)`

        Parameters
        ----------
        buf: legate-store-like (1-dimensional)
            A Legate store or any object implementing `__legate_data_interface__` to
            write into buffer.
        """
        assert not self._closed
        if "w" not in self._flags and "+" not in self._flags:
            raise ValueError(f"Cannot write to a file opened with flags={self._flags}")

        input = get_legate_store(buf)
        task = context.create_auto_task(TaskOpCode.WRITE)
        task.add_scalar_arg(self._filepath, types.string)
        task.add_input(input)
        task.set_side_effect(True)
        task.execute()
