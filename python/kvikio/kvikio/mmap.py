# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import stat
from typing import Any, Optional

from kvikio._lib.mmap import InternalMmapHandle
from kvikio.cufile import IOFuture


class Mmap:
    """Handle of a memory-mapped file"""

    def __init__(
        self,
        file_path: os.PathLike,
        flags: str = "r",
        initial_map_size: Optional[int] = None,
        initial_map_offset: int = 0,
        mode: int = stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IROTH,
        map_flags: Optional[int] = None,
    ):
        """Construct a new memory-mapped file handle

        Parameters
        ----------
        file_path : os.PathLike
            File path.
        flags : str, optional

            - ``r``: Open for reading (default)
            - ``w``: (Not implemented yet) Open for writing, truncating the file first
            - ``a``: (Not implemented yet) Open for writing, appending to the end of
              file if it exists
            - ``+``: (Not implemented yet) Open for updating (reading and writing)
        initial_map_size : int, optional
            Size in bytes of the mapped region. If not specified, map the region
            starting from ``initial_map_offset`` to the end of file.
        initial_map_offset : int, optional
            File offset of the mapped region. Default is 0.
        mode : int, optional
            Access mode (permissions) to use if creating a new file. Default is
            0644 (octal), 420 (decimal).
        map_flags : int, optional
            Flags to be passed to the system call ``mmap``. See `mmap(2)` for details.
        """
        self._handle = InternalMmapHandle(
            file_path, flags, initial_map_size, initial_map_offset, mode, map_flags
        )

    def initial_map_size(self) -> int:
        """Size in bytes of the mapped region when the mapping handle was constructed

        Returns
        -------
        int
            Initial size of the mapped region.
        """
        return self._handle.initial_map_size()

    def initial_map_offset(self) -> int:
        """File offset of the mapped region when the mapping handle was constructed

        Returns
        -------
        int
            Initial file offset of the mapped region.
        """
        return self._handle.initial_map_offset()

    def file_size(self) -> int:
        """Get the file size if the file is open

        Returns 0 if the file is closed.

        Returns
        -------
        int
            The file size in bytes.
        """
        return self._handle.file_size()

    def close(self) -> None:
        """Close the mapping handle if it is open; do nothing otherwise

        Unmaps the memory region and closes the underlying file descriptor.
        """
        self._handle.close()

    def closed(self) -> bool:
        """Whether the mapping handle is closed

        Returns
        -------
        bool
            Boolean answer.
        """
        return self._handle.closed()

    def read(self, buf: Any, size: Optional[int] = None, offset: int = 0) -> int:
        """Sequential read ``size`` bytes from the file to the destination buffer
        ``buf``

        Parameters
        ----------
        buf : buffer-like or array-like
            Address of the host or device memory (destination buffer).
        size : int, optional
            Size in bytes to read. If not specified, read starts from ``offset``
            to the end of file.
        offset : int, optional
            File offset. Default is 0.

        Returns
        -------
        int
            Number of bytes that have been read.

        Raises
        ------
        IndexError
            If the read region specified by ``offset`` and ``size`` is outside the
            initial region specified when the mapping handle was constructed.
        RuntimeError
            If the mapping handle is closed.
        """
        return self._handle.read(buf, size, offset)

    def pread(
        self,
        buf: Any,
        size: Optional[int] = None,
        offset: int = 0,
        task_size: Optional[int] = None,
    ) -> IOFuture:
        """Parallel read ``size`` bytes from the file to the destination buffer ``buf``

        Parameters
        ----------
        buf : buffer-like or array-like
            Address of the host or device memory (destination buffer).
        size : int, optional
            Size in bytes to read. If not specified, read starts from ``offset``
            to the end of file.
        offset : int, optional
            File offset. Default is 0.
        task_size : int, optional
            Size of each task in bytes for parallel execution. If None, uses
            the default task size from :func:`kvikio.defaults.task_size`.

        Returns
        -------
        IOFuture
            Future that on completion returns the size of bytes that were successfully
            read.

        Raises
        ------
        IndexError
            If the read region specified by ``offset`` and ``size`` is outside the
            initial region specified when the mapping handle was constructed.
        RuntimeError
            If the mapping handle is closed.

        Notes
        -----
        The returned IOFuture object's ``get()`` should not be called after the lifetime
        of the MmapHandle object ends. Otherwise, the behavior is undefined.
        """
        return IOFuture(self._handle.pread(buf, size, offset, task_size))
