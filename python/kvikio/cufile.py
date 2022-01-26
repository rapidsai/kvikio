# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.

import pathlib
from typing import Union

from ._lib import libkvikio  # type: ignore


class IOFuture:
    """Future for CuFile IO"""

    __slots__ = "_handle"

    def __init__(self, handle) -> None:
        self._handle = handle

    def get(self) -> int:
        return self._handle.get()


class CuFile:
    """ File handle for GPUDirect Storage (GDS) """

    def __init__(self, file: Union[pathlib.Path, str], flags: str = "r"):
        """Open and register file for GDS IO operations

        The file is always opened in binary and direct mode.

        Parameters
        ----------
        file: pathlib.Path or str
            Path-like object giving the pathname (absolute or relative to the current
            working directory) of the file to be opened and registered.
        flags: str, optional
            "r" -> "open for reading (default)"
            "w" -> "open for writing, truncating the file first"
            "a" -> "open for writing, appending to the end of file if it exists"
            "+" -> "open for updating (reading and writing)"
        """
        self._handle = libkvikio.CuFile(file, flags)

    def close(self) -> None:
        self._handle.close()

    @property
    def closed(self) -> bool:
        return self._handle.closed()

    def fileno(self) -> int:
        return self._handle.fileno()

    def open_flags(self) -> int:
        return self._handle.open_flags()

    def __enter__(self) -> "CuFile":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def pread(
        self, buf, size: int = None, file_offset: int = 0, ntasks=None
    ) -> IOFuture:
        return IOFuture(self._handle.pread(buf, size, file_offset, ntasks))

    def pwrite(
        self, buf, size: int = None, file_offset: int = 0, ntasks=None
    ) -> IOFuture:
        return IOFuture(self._handle.pwrite(buf, size, file_offset, ntasks))

    def read(self, buf, size: int = None, file_offset: int = 0, ntasks=None) -> int:
        return self.pread(buf, size, file_offset, ntasks).get()

    def write(self, buf, size: int = None, file_offset: int = 0, ntasks=None) -> int:
        return self.pwrite(buf, size, file_offset, ntasks).get()
