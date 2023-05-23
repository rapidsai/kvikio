# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.

import os
from enum import IntEnum
from typing import Any

from legate.core import Library, get_legate_runtime
from legate_kvikio.install_info import header, libpath


class LibraryDescription(Library):
    def __init__(self) -> None:
        self.shared_object: Any = None

    @property
    def cffi(self) -> Any:
        return self.shared_object

    def get_name(self) -> str:
        return "legate_kvikio"

    def get_shared_library(self) -> str:
        return os.path.join(libpath, f"liblegate_kvikio{self.get_library_extension()}")

    def get_c_header(self) -> str:
        return header

    def get_registration_callback(self) -> str:
        return "legate_kvikio_perform_registration"

    def initialize(self, shared_object: Any) -> None:
        self.shared_object = shared_object

    def destroy(self) -> None:
        pass


description = LibraryDescription()
context = get_legate_runtime().register_library(description)


class TaskOpCode(IntEnum):
    WRITE = description.cffi.OP_WRITE
    READ = description.cffi.OP_READ
    TILE_WRITE = description.cffi.OP_TILE_WRITE
    TILE_READ = description.cffi.OP_TILE_READ
    TILE_READ_BY_OFFSETS = description.cffi.OP_TILE_READ_BY_OFFSETS
