# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.

import pathlib
from typing import Union

from ._lib import libkvikio  # type: ignore


class IOFuture:
    """Future for CuFile IO

    This class shouldn't be used directly, instead non-blocking IO operations such
    as `CuFile.pread` and `CuFile.pwrite` returns an instance of this class. Use
    `.get()` to wait on the completion of the IO operation and retrieve the result.
    """

    __slots__ = "_handle"

    def __init__(self, handle):
        self._handle = handle

    def get(self) -> int:
        """Retrieve the result of the IO operation that created this future

        This call blocks until the IO operation finishes.

        Returns
        ------
        int
            The size of bytes that were read or written successfully.
        """
        return self._handle.get()

    def done(self) -> bool:
        """Return True if the future is done.

        Returns
        ------
        bool
            Whether the future is done or not
        """
        return self._handle.done()


class CuFile:
    """File handle for GPUDirect Storage (GDS)"""

    def __init__(self, file: Union[pathlib.Path, str], flags: str = "r"):
        """Open and register file for GDS IO operations

        CuFile opens the file twice and maintains two file descriptors.
        One file is opened with the specified `flags` and the other file is
        opened with the `flags` plus the `O_DIRECT` flag.

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
        """Deregister the file and close the file"""
        self._handle.close()

    @property
    def closed(self) -> bool:
        return self._handle.closed()

    def fileno(self) -> int:
        """Get the file descripter of the open file"""
        return self._handle.fileno()

    def open_flags(self) -> int:
        """Get the flags of the file descripter (see open(2))"""
        return self._handle.open_flags()

    def __enter__(self) -> "CuFile":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def pread(
        self, buf, size: int = None, file_offset: int = 0, task_size=None
    ) -> IOFuture:
        """Reads specified bytes from the file into device or host memory in parallel

        `pread` reads the data from a specified file at a specified offset and size
        bytes into `buf`. The API works correctly for unaligned offsets and any data
        size, although the performance might not match the performance of aligned reads.

        `pread` is non-blocking and returns a `IOFuture` that can be waited upon. It
        partitions the operation into tasks of size `task_size` for execution in the
        default thread pool.

        Parameters
        ----------
        buf: buffer-like or array-like
            Device or host buffer to read into.
        size: int
            Size in bytes to read.
        file_offset: int, optional
            Offset in the file to read from.
        task_size: int, default=kvikio.defaults.task_size()
            Size of each task in bytes.

        Returns
        ------
        IOFuture
            Future that on completion returns the size of bytes that were successfully
            read.
        """
        return IOFuture(self._handle.pread(buf, size, file_offset, task_size))

    def pwrite(
        self, buf, size: int = None, file_offset: int = 0, task_size=None
    ) -> IOFuture:
        """Writes specified bytes from device or host memory into the file in parallel

        `pwrite` writes the data from `buf` to the file at a specified offset and size
        bytes. The API works correctly for unaligned offset and data sizes, although
        the performance might not match the performance of aligned writes.

        `pwrite` is non-blocking and returns a `IOFuture` that can be waited upon. It
        partitions the operation into tasks of size `task_size` for execution in the
        default thread pool.

        Parameters
        ----------
        buf: buffer-like or array-like
            Device or host buffer to write to.
        size: int
            Size in bytes to read.
        file_offset: int, optional
            Offset in the file to write from.
        task_size: int, default=kvikio.defaults.task_size()
            Size of each task in bytes.

        Returns
        ------
        IOFuture
            Future that on completion returns the size of bytes that were successfully
            written.
        """
        return IOFuture(self._handle.pwrite(buf, size, file_offset, task_size))

    def read(self, buf, size: int = None, file_offset: int = 0, task_size=None) -> int:
        """Reads specified bytes from the file into the device memory in parallel

        This is a blocking version of `.pread`.

        Parameters
        ----------
        buf: buffer-like or array-like
            Device buffer to read into.
        size: int
            Size in bytes to read.
        file_offset: int, optional
            Offset in the file to read from.
        task_size: int, default=kvikio.defaults.task_size()
            Size of each task in bytes.

        Returns
        ------
        int
            The size of bytes that were successfully read.
        """
        return self.pread(buf, size, file_offset, task_size).get()

    def write(self, buf, size: int = None, file_offset: int = 0, task_size=None) -> int:
        """Writes specified bytes from the device memory into the file in parallel

        This is a blocking version of `.pwrite`.

        Parameters
        ----------
        buf: buffer-like or array-like
            Device buffer to write to.
        size: int
            Size in bytes to read.
        file_offset: int, optional
            Offset in the file to write from.
        task_size: int, default=kvikio.defaults.task_size()
            Size of each task in bytes.

        Returns
        ------
        int
            The size of bytes that were successfully written.
        """
        return self.pwrite(buf, size, file_offset, task_size).get()

    def raw_read(
        self, buf, size: int = None, file_offset: int = 0, dev_offset: int = 0
    ) -> int:
        """Reads specified bytes from the file into the device memory

        This is a low-level version of `.read` that doesn't use threads and
        does not support host memory.

        Parameters
        ----------
        buf: buffer-like or array-like
            Device buffer to read into.
        size: int
            Size in bytes to read.
        file_offset: int, optional
            Offset in the file to read from.
        dev_offset: int, optional
            Offset in the `buf` to read from.

        Returns
        ------
        int
            The size of bytes that were successfully read.
        """
        return self._handle.read(buf, size, file_offset, dev_offset)

    def raw_write(
        self, buf, size: int = None, file_offset: int = 0, dev_offset: int = 0
    ) -> int:
        """Writes specified bytes from the device memory into the file

        This is a low-level version of `.write` that doesn't use threads and
        does not support host memory.

        Parameters
        ----------
        buf: buffer-like or array-like
            Device buffer to write to.
        size: int
            Size in bytes to read.
        file_offset: int, optional
            Offset in the file to write from.
        dev_offset: int, optional
            Offset in the `buf` to write from.

        Returns
        ------
        int
            The size of bytes that were successfully written.
        """
        return self._handle.write(buf, size, file_offset, dev_offset)
