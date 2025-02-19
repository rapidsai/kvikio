# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.

import pathlib

import cupy as cp
import pytest
import zarr.core.buffer
import zarr.storage
from zarr.core.buffer.cpu import Buffer
from zarr.testing.store import StoreTests

import kvikio.zarr_v3


@pytest.mark.asyncio
async def test_basic(tmp_path: pathlib.Path) -> None:
    src = zarr.storage.LocalStore(tmp_path)
    arr = zarr.create_array(src, name="a", shape=(10,), dtype="u4", zarr_format=3)
    arr[:5] = 0
    arr[5:] = 1

    assert await src.exists("a/zarr.json")
    store = kvikio.zarr_v3.GDSStore(tmp_path, read_only=True)
    assert await store.exists("a/zarr.json")

    # regular works
    zarr.open_array(src, path="a", zarr_format=3)

    # read
    with zarr.config.enable_gpu():
        arr = zarr.open_array(store, path="a", zarr_format=3)
        result = arr[:]
        assert isinstance(result, cp.ndarray)
        expected = cp.array([0] * 5 + [1] * 5, dtype="uint32")
        cp.testing.assert_array_equal(result, expected)


class TestKvikIOStore(StoreTests[kvikio.zarr_v3.GDSStore, Buffer]):
    store_cls = kvikio.zarr_v3.GDSStore
    buffer_cls = Buffer

    async def get(self, store: kvikio.zarr_v3.GDSStore, key: str) -> Buffer:
        return self.buffer_cls.from_bytes((store.root / key).read_bytes())

    async def set(
        self, store: kvikio.zarr_v3.GDSStore, key: str, value: Buffer
    ) -> None:
        parent = (store.root / key).parent
        if not parent.exists():
            parent.mkdir(parents=True)
        (store.root / key).write_bytes(value.to_bytes())

    @pytest.fixture
    def store_kwargs(self, tmpdir: pathlib.Path) -> dict[str, str]:
        kwargs = {"root": str(tmpdir)}
        return kwargs

    @pytest.fixture
    async def store(self, store_kwargs: dict[str, str]) -> kvikio.zarr_v3.GDSStore:
        return self.store_cls(**store_kwargs)

    @pytest.fixture
    async def store_not_open(
        self, store_kwargs: dict[str, str]
    ) -> kvikio.zarr_v3.GDSStore:
        return self.store_cls(**store_kwargs)
