# SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import io
import os
import pathlib
from typing import Optional, Union

from kvikio._lib import file_handle  # type: ignore
from kvikio.utils import kvikio_deprecation_notice


class IOFutureStream:
    """Future for CuFile async IO

    This class shouldn't be used directly, instead non-blocking async IO operations
    such as `CuFile.raw_read_async` and `CuFile.raw_write_async` returns an instance
    of this class.

    The instance must be kept alive alive until all data has been read from disk. One
    way to do this, is by calling `StreamFuture.check_bytes_done()`, which will
    synchronize the associated stream and return the number of bytes read.
    """

    __slots__ = "_handle"

    def __init__(self, handle):
        self._handle = handle

    def check_bytes_done(self) -> int:
        return self._handle.check_bytes_done()


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
        -------
        int
            The size of bytes that were read or written successfully.
        """
        return self._handle.get()

    def done(self) -> bool:
        """Return True if the future is done.

        Returns
        -------
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
        self._handle = file_handle.CuFile(file, flags)

    def close(self) -> None:
        """Deregister the file and close the file"""
        self._handle.close()

    @property
    def closed(self) -> bool:
        return self._handle.closed()

    def fileno(self) -> int:
        """Get the file descriptor of the open file"""
        return self._handle.fileno()

    def open_flags(self) -> int:
        """Get the flags of the file descriptor (see open(2))"""
        return self._handle.open_flags()

    def __enter__(self) -> "CuFile":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def pread(
        self,
        buf,
        size: Optional[int] = None,
        file_offset: int = 0,
        task_size: Optional[int] = None,
    ) -> IOFuture:
        """Reads specified bytes from the file into device or host memory in parallel

        `pread` reads the data from a specified file at a specified offset and size
        bytes into `buf`. The API works correctly for unaligned offsets and any data
        size, although the performance might not match the performance of aligned reads.
        See additional details in the notes below.

        `pread` is non-blocking and returns a `IOFuture` that can be waited upon. It
        partitions the operation into tasks of size `task_size` for execution in the
        default thread pool.

        Parameters
        ----------
        buf: buffer-like or array-like
            Device or host buffer to read into.
        size: int, optional
            Size in bytes to read.
        file_offset: int, optional
            Offset in the file to read from.
        task_size: int, default=kvikio.defaults.task_size()
            Size of each task in bytes.

        Returns
        -------
        IOFuture
            Future that on completion returns the size of bytes that were successfully
            read.

        Notes
        -----
        KvikIO can only make use of GDS for reads that are aligned to a page boundary.
        For unaligned reads, KvikIO has to split the reads into aligned and unaligned
        parts. The GPU page size is 4kB, so all reads should be at an offset that is a
        multiple of 4096 bytes. If the desired `file_offset` is not a multiple of 4096,
        it is likely desirable to round down to the nearest multiple of 4096 and discard
        any undesired bytes from the resulting data. Similarly, it is optimal for `size`
        to be a multiple of 4096 bytes. When GDS isn't used, this is less critical.
        """
        return IOFuture(self._handle.pread(buf, size, file_offset, task_size))

    def pwrite(
        self,
        buf,
        size: Optional[int] = None,
        file_offset: int = 0,
        task_size: Optional[int] = None,
    ) -> IOFuture:
        """Writes specified bytes from device or host memory into the file in parallel

        `pwrite` writes the data from `buf` to the file at a specified offset and size.
        The API works correctly for unaligned offset and data sizes, although the
        performance is not on-par with aligned writes. See additional details in the
        notes below.

        `pwrite` is non-blocking and returns a `IOFuture` that can be waited upon. It
        partitions the operation into tasks of size `task_size` for execution in the
        default thread pool.

        Parameters
        ----------
        buf: buffer-like or array-like
            Device or host buffer to write to.
        size: int, optional
            Size in bytes to write.
        file_offset: int, optional
            Offset in the file to write from.
        task_size: int, default=kvikio.defaults.task_size()
            Size of each task in bytes.

        Returns
        -------
        IOFuture
            Future that on completion returns the size of bytes that were successfully
            written.

        Notes
        -----
        KvikIO can only make use of GDS for writes that are aligned to a page boundary.
        For unaligned writes, KvikIO has to split the writes into aligned and unaligned
        parts. The GPU page size is 4kB, so all writes should be at an offset that is a
        multiple of 4096 bytes. If the desired `file_offset` is not a multiple of 4096,
        it is likely desirable to round down to the nearest multiple of 4096 and discard
        any undesired bytes from the resulting data. Similarly, it is optimal for `size`
        to be a multiple of 4096 bytes. When GDS isn't used, this is less critical.
        """
        return IOFuture(self._handle.pwrite(buf, size, file_offset, task_size))

    def read(
        self,
        buf,
        size: Optional[int] = None,
        file_offset: int = 0,
        task_size: Optional[int] = None,
    ) -> int:
        """Reads specified bytes from the file into the device memory in parallel

        This is a blocking version of `.pread`.

        Parameters
        ----------
        buf: buffer-like or array-like
            Device buffer to read into.
        size: int, optional
            Size in bytes to read.
        file_offset: int, optional
            Offset in the file to read from.
        task_size: int, default=kvikio.defaults.task_size()
            Size of each task in bytes.

        Returns
        -------
        int
            The size of bytes that were successfully read.

        Notes
        -----
        KvikIO can only make use of GDS for reads that are aligned to a page boundary.
        For unaligned reads, KvikIO has to split the reads into aligned and unaligned
        parts. The GPU page size is 4kB, so all reads should be at an offset that is a
        multiple of 4096 bytes. If the desired `file_offset` is not a multiple of 4096,
        it is likely desirable to round down to the nearest multiple of 4096 and discard
        any undesired bytes from the resulting data. Similarly, it is optimal for `size`
        to be a multiple of 4096 bytes. When GDS isn't used, this is less critical.
        """
        return self.pread(buf, size, file_offset, task_size).get()

    def write(
        self,
        buf,
        size: Optional[int] = None,
        file_offset: int = 0,
        task_size: Optional[int] = None,
    ) -> int:
        """Writes specified bytes from the device memory into the file in parallel

        This is a blocking version of `.pwrite`.

        Parameters
        ----------
        buf: buffer-like or array-like
            Device buffer to write to.
        size: int, optional
            Size in bytes to write.
        file_offset: int, optional
            Offset in the file to write from.
        task_size: int, default=kvikio.defaults.task_size()
            Size of each task in bytes.

        Returns
        -------
        int
            The size of bytes that were successfully written.

        Notes
        -----
        KvikIO can only make use of GDS for writes that are aligned to a page boundary.
        For unaligned writes, KvikIO has to split the writes into aligned and unaligned
        parts. The GPU page size is 4kB, so all writes should be at an offset that is a
        multiple of 4096 bytes. If the desired `file_offset` is not a multiple of 4096,
        it is likely desirable to round down to the nearest multiple of 4096 and discard
        any undesired bytes from the resulting data. Similarly, it is optimal for `size`
        to be a multiple of 4096 bytes. When GDS isn't used, this is less critical.
        """
        return self.pwrite(buf, size, file_offset, task_size).get()

    def raw_read_async(
        self,
        buf,
        raw_stream: int,
        size: Optional[int] = None,
        file_offset: int = 0,
        dev_offset: int = 0,
    ) -> IOFutureStream:
        """Reads specified bytes from the file into the device memory asynchronously

        This is an async version of `.raw_read` that doesn't use threads and
        does not support host memory.

        Parameters
        ----------
        buf: buffer-like or array-like
            Device buffer to read into.
        raw_stream: int
            Raw CUDA stream to perform the read operation asynchronously.
        size: int, optional
            Size in bytes to read.
        file_offset: int, optional
            Offset in the file to read from.

        Returns
        -------
        IOFutureStream
            Future that when executed ".check_bytes_done()" returns the size of bytes
            that were successfully read. The instance must be kept alive until
            all data has been read from disk. One way to do this, is by calling
            `IOFutureStream.check_bytes_done()`, which will synchronize the associated
            stream and return the number of bytes read.
        """
        return self._handle.read_async(buf, size, file_offset, dev_offset, raw_stream)

    def raw_write_async(
        self,
        buf,
        raw_stream: int,
        size: Optional[int] = None,
        file_offset: int = 0,
        dev_offset: int = 0,
    ) -> IOFutureStream:
        """Writes specified bytes from the device memory into the file asynchronously

        This is an async version of `.raw_write` that doesn't use threads and
        does not support host memory.

        Parameters
        ----------
        buf: buffer-like or array-like
            Device buffer to write to.
        raw_stream: int
            Raw CUDA stream to perform the write operation asynchronously.
        size: int, optional
            Size in bytes to write.
        file_offset: int, optional
            Offset in the file to write from.

        Returns
        -------
        IOFutureStream
            Future that when executed ".check_bytes_done()" returns the size of bytes
            that were successfully written. The instance must be kept alive until
            all data has been written to disk. One way to do this, is by calling
            `IOFutureStream.check_bytes_done()`, which will synchronize the associated
            stream and return the number of bytes written.
        """
        return self._handle.write_async(buf, size, file_offset, dev_offset, raw_stream)

    def raw_read(
        self,
        buf,
        size: Optional[int] = None,
        file_offset: int = 0,
        dev_offset: int = 0,
    ) -> int:
        """Reads specified bytes from the file into the device memory

        This is a low-level version of `.read` that doesn't use threads and
        does not support host memory.

        Parameters
        ----------
        buf: buffer-like or array-like
            Device buffer to read into.
        size: int, optional
            Size in bytes to read.
        file_offset: int, optional
            Offset in the file to read from.
        dev_offset: int, optional
            Offset in the `buf` to read from.

        Returns
        -------
        int
            The size of bytes that were successfully read.

        Notes
        -----
        KvikIO can only make use of GDS for reads that are aligned to a page boundary.
        For unaligned reads, KvikIO has to split the reads into aligned and unaligned
        parts. The GPU page size is 4kB, so all reads should be at an offset that is a
        multiple of 4096 bytes. If the desired `file_offset` is not a multiple of 4096,
        it is likely desirable to round down to the nearest multiple of 4096 and discard
        any undesired bytes from the resulting data. Similarly, it is optimal for `size`
        to be a multiple of 4096 bytes. When GDS isn't used, this is less critical.
        """
        return self._handle.read(buf, size, file_offset, dev_offset)

    def raw_write(
        self,
        buf,
        size: Optional[int] = None,
        file_offset: int = 0,
        dev_offset: int = 0,
    ) -> int:
        """Writes specified bytes from the device memory into the file

        This is a low-level version of `.write` that doesn't use threads and
        does not support host memory.

        Parameters
        ----------
        buf: buffer-like or array-like
            Device buffer to write to.
        size: int, optional
            Size in bytes to write.
        file_offset: int, optional
            Offset in the file to write from.
        dev_offset: int, optional
            Offset in the `buf` to write from.

        Returns
        -------
        int
            The size of bytes that were successfully written.

        Notes
        -----
        KvikIO can only make use of GDS for writes that are aligned to a page boundary.
        For unaligned writes, KvikIO has to split the writes into aligned and unaligned
        parts. The GPU page size is 4kB, so all writes should be at an offset that is a
        multiple of 4096 bytes. If the desired `file_offset` is not a multiple of 4096,
        it is likely desirable to round down to the nearest multiple of 4096 and discard
        any undesired bytes from the resulting data. Similarly, it is optimal for `size`
        to be a multiple of 4096 bytes. When GDS isn't used, this is less critical.
        """
        return self._handle.write(buf, size, file_offset, dev_offset)

    def is_direct_io_supported(self) -> bool:
        """Whether Direct I/O is supported on this file handle.

        This is determined by two factors:
        - Direct I/O support from the operating system and the file system
        - KvikIO global setting `auto_direct_io_read` and `auto_direct_io_write`. If
        both values are false, Direct I/O will not be supported on this file handle.

        Returns
        -------
        bool
            Whether Direct I/O is supported
        """
        return self._handle.is_direct_io_supported()


def get_page_cache_info(
    file: Union[os.PathLike, str, int, io.IOBase],
) -> tuple[int, int]:
    """Obtain the page cache residency information for a given file

    Example:

    .. code-block:: python

       pages_cached, pages_total = kvikio.get_page_cache_info(my_file)
       fraction_cached = pages_cached / pages_total

    Parameters
    ----------
    file: a path-like object, or string, or file descriptor, or file object
        File to check.

    Returns
    -------
    tuple[int, int]
        A pair containing the number of pages resident in the page cache
        and the total number of pages.
    """
    return file_handle.get_page_cache_info(file)


def drop_file_page_cache(
    file: Union[os.PathLike, str, int, io.IOBase],
    offset: int = 0,
    length: int = 0,
    sync_first: bool = True,
) -> None:
    """Drop page cache for a specific file.

    Advises the kernel to evict cached pages for the specified file descriptor using
    `posix_fadvise` with `POSIX_FADV_DONTNEED`.

    Parameters
    ----------
    file: a path-like object, or string, or file descriptor, or file object
        File to operate on.
    offset: int
        Starting byte offset (default: 0 for beginning of file)
    length: int
        Number of bytes to drop (default: 0, meaning entire file from offset)
    sync_first: bool
        Whether to flush dirty pages to disk before dropping. If `True`, `fdatasync`
        will be called prior to dropping. This ensures dirty pages become clean and
        thus droppable. Can be set to `False` if we are certain no dirty pages exist
        for this file.

    Notes
    -----
    - This is the preferred method for benchmark cache invalidation as it:

      - Requires no elevated privileges
      - Affects only the specified file, not other processes
      - Has minimal overhead (no child process spawned)

    - The page cache dropping takes place in granularity of full pages. If the
      specified range does not align to page boundaries, partial pages at the start
      and end of the range are retained. Only pages fully contained within the range
      are dropped.

    - For dropping page cache system-wide (requires elevated privileges), see
      :func:`drop_system_page_cache()`.
    """
    return file_handle.drop_file_page_cache(file, offset, length, sync_first)


def drop_system_page_cache(
    reclaim_dentries_and_inodes: bool = True, sync_first: bool = True
) -> bool:
    """Drop the system page cache.

    Parameters
    ----------
    reclaim_dentries_and_inodes: bool
        Whether to free reclaimable slab objects which include dentries and inodes.

        - If `True`, equivalent to executing `/sbin/sysctl vm.drop_caches=3`;
        - If `False`, equivalent to executing `/sbin/sysctl vm.drop_caches=1`.
    sync_first: bool
        Whether to flush dirty pages to disk before dropping. If `True`, `sync` will be
        called prior to dropping. This ensures dirty pages become clean and thus
        droppable.

    Returns
    -------
    bool
       Whether the page cache has been successfully dropped

    Notes
    -----
    - This drops page cache system-wide, affecting all processes. For dropping cache
      for a specific file without elevated privileges, see :func:`drop_file_page_cache`.
    - This function creates a child process and executes the cache dropping shell
      command in the following order:

      - Execute the command without `sudo` prefix. This is for the superuser and also
        for specially configured systems where unprivileged users cannot execute
        `/usr/bin/sudo` but can execute `/sbin/sysctl`. If this step succeeds, the
        function returns `True` immediately.
      - Execute the command with `sudo` prefix. This is for the general case where
        selective unprivileged users have permission to run `/sbin/sysctl` with `sudo`
        prefix.
    """
    return file_handle.drop_system_page_cache(reclaim_dentries_and_inodes, sync_first)


@kvikio_deprecation_notice(
    "Use :func:`drop_system_page_cache` instead.",
    # rapids-pre-commit-hooks: disable-next-line[verify-hardcoded-version]
    since="26.04",
)
def clear_page_cache(
    reclaim_dentries_and_inodes: bool = True, clear_dirty_pages: bool = True
) -> bool:
    """Drop the system page cache. Deprecated. Use :func:`drop_system_page_cache` instead."""
    return file_handle.drop_system_page_cache(
        reclaim_dentries_and_inodes, clear_dirty_pages
    )
