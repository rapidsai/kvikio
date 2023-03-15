# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.

from typing import Any

from legate.core import Store


def get_legate_store(buf: Any) -> Store:
    """Extracts a Legate store from object

    Supports any object that implements the legate data interface
    (`__legate_data_interface__`).

    Parameters
    ----------
    buf: legate-store-like
        Object implement the Legate store interface

    Returns
    -------
    Store
        The extracted Legate store
    """
    if isinstance(buf, Store):
        return buf
    data = buf.__legate_data_interface__["data"]
    field = next(iter(data))
    array = data[field]
    _, store = array.stores()
    return store
