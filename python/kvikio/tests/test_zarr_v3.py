# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pathlib

import cupy as cp
import pytest

pytest.importorskip("zarr", minversion="3.0.0")

# these must follow the pytest.importorskip

import zarr.core.buffer  # noqa: E402
import zarr.storage  # noqa: E402
from zarr.core.buffer.gpu import Buffer  # noqa: E402
from zarr.testing.store import StoreTests  # noqa: E402

import kvikio.zarr  # noqa: E402


@pytest.mark.asyncio
async def test_basic(tmp_path: pathlib.Path) -> None:
    src = zarr.storage.LocalStore(tmp_path)
    arr = zarr.create_array(src, name="a", shape=(10,), dtype="u4", zarr_format=3)
    arr[:5] = 0
    arr[5:] = 1

    assert await src.exists("a/zarr.json")
    store = kvikio.zarr.GDSStore(tmp_path, read_only=True)
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


# These tests use zarr-python's StoreTests
# https://zarr.readthedocs.io/en/stable/api/zarr/testing/store/index.html#zarr.testing.store.StoreTests
# which provide a set of unit tests each implementation is expected to pass.


class TestKvikIOStore(StoreTests[kvikio.zarr.GDSStore, Buffer]):
    store_cls = kvikio.zarr.GDSStore
    buffer_cls = Buffer

    async def get(self, store: kvikio.zarr.GDSStore, key: str) -> Buffer:
        return self.buffer_cls.from_bytes((store.root / key).read_bytes())

    async def set(
        self,
        store: kvikio.zarr.GDSStore,
        key: str,
        value: Buffer,  # type: ignore[override]
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
    async def store(self, store_kwargs: dict[str, str]) -> kvikio.zarr.GDSStore:
        # ignore Argument 1 has incompatible type "**Dict[str, str]"; expected "bool"
        return self.store_cls(**store_kwargs)  # type: ignore[arg-type]

    @pytest.fixture
    async def store_not_open(
        self, store_kwargs: dict[str, str]
    ) -> kvikio.zarr.GDSStore:
        # ignore Argument 1 has incompatible type "**Dict[str, str]"; expected "bool"
        return self.store_cls(**store_kwargs)  # type: ignore[arg-type]
