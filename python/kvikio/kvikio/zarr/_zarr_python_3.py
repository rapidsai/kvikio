# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import functools
import os
from pathlib import Path

import packaging.version
import zarr

_zarr_version = packaging.version.parse(zarr.__version__)

if _zarr_version < packaging.version.parse("3.0.0"):
    # We include this runtime package checking to help users who relied on
    # installing kvikio to also include zarr, which is not an optional dependency.
    raise ImportError(
        f"'zarr>=3' is required, but 'zarr=={_zarr_version}' is installed."
    )

import zarr.storage  # noqa: E402
from zarr.abc.store import (  # noqa: E402
    ByteRequest,
    OffsetByteRequest,
    RangeByteRequest,
    SuffixByteRequest,
)
from zarr.core.buffer import Buffer, BufferPrototype  # noqa: E402
from zarr.core.buffer.core import default_buffer_prototype  # noqa: E402

import kvikio  # noqa: E402

# The GDSStore implementation follows the `LocalStore` implementation
# at https://github.com/zarr-developers/zarr-python/blob/main/src/zarr/storage/_local.py
# with differences coming swapping in `cuFile` for the stdlib open file object.


@functools.cache
def _is_ge_zarr_3_0_7():
    return _zarr_version >= packaging.version.parse("3.0.7")


def _get(
    path: Path, prototype: BufferPrototype, byte_range: ByteRequest | None
) -> Buffer:
    file_size = os.path.getsize(path)
    file_offset: int

    match byte_range:
        case None:
            nbytes = file_size
            file_offset = 0
        case OffsetByteRequest():
            nbytes = max(0, file_size - byte_range.offset)
            file_offset = byte_range.offset
        case RangeByteRequest():
            nbytes = byte_range.end - byte_range.start
            file_offset = byte_range.start
        case SuffixByteRequest():
            nbytes = byte_range.suffix
            file_offset = max(0, file_size - byte_range.suffix)
        case _:
            # This isn't allowed by mypy, but the tests assert we raise
            # something here.
            raise TypeError(f"Unexpected byte_range, got {byte_range}")

    # kvikio doesn't support reading past the end of a file. Some zarr tests
    # rely on this behavior: to "read" 3 bytes out of a 0 byte file, or to
    # "seek" past the end of a file with file_offset. The semantics seem to
    # be roughly the same as slicing an empty bytestring.

    nbytes = min(nbytes, file_size)
    file_offset = min(file_offset, file_size)

    if _is_ge_zarr_3_0_7():
        dtype = "B"
    else:
        dtype = "b"

    raw = prototype.nd_buffer.create(shape=(nbytes,), dtype=dtype).as_ndarray_like()
    buf = prototype.buffer.from_array_like(raw)

    with kvikio.CuFile(path) as f:
        # Note: this currently creates an IOFuture and then blocks
        # on reading it. The blocking read means this is in a regular
        # sync function, and so this must be run in a threadpool.
        future = f.pread(raw, size=nbytes, file_offset=file_offset)
        future.get()  # blocks

    return buf


def _put(
    path: Path,
    value: Buffer,
    start: int | None = None,
    exclusive: bool = False,
) -> int | None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if start is not None:
        with kvikio.CuFile(path, "r+b") as f:
            f.write(value.as_array_like(), file_offset=start)
        return None
    else:
        buf = value.as_array_like()
        if exclusive:
            if path.exists():
                raise FileExistsError(f"File exists: {path}")
        mode = "wb"
        with kvikio.CuFile(path, flags=mode) as f:
            return f.write(buf)


class GDSStore(zarr.storage.LocalStore):
    def __repr__(self) -> str:
        return f"{type(self).__name__}('{self}')"

    async def get(
        self,
        key: str,
        prototype: BufferPrototype | None = None,
        byte_range: ByteRequest | None = None,
    ) -> Buffer | None:
        if prototype is None:
            prototype = default_buffer_prototype()
        if not self._is_open:
            await self._open()
        assert isinstance(key, str)
        path = self.root / key

        try:
            return await asyncio.to_thread(_get, path, prototype, byte_range)
        except (FileNotFoundError, IsADirectoryError, NotADirectoryError):
            return None

    async def set(self, key: str, value: Buffer) -> None:
        return await self._set(key, value)

    async def _set(self, key: str, value: Buffer, exclusive: bool = False) -> None:
        if not self._is_open:
            await self._open()
        self._check_writable()
        assert isinstance(key, str)
        if not isinstance(value, Buffer):
            raise TypeError(
                f"LocalStore.set(): `value` must be a Buffer instance. Got an "
                f"instance of {type(value)} instead."
            )
        path = self.root / key

        await asyncio.to_thread(_put, path, value, start=None, exclusive=exclusive)
