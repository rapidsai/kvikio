# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.

import os
from typing import Any

from legate_kvikio.install_info import header, libpath

from legate.core import Library, ResourceConfig, get_legate_runtime


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

    def get_resource_configuration(self) -> ResourceConfig:
        assert self.shared_object is not None
        config = ResourceConfig()
        config.max_mappers = 1
        config.max_reduction_ops = 8
        config.max_projections = 0
        config.max_shardings = 0
        return config

    def initialize(self, shared_object: Any) -> None:
        self.shared_object = shared_object

    def destroy(self) -> None:
        pass


description = LibraryDescription()
context = get_legate_runtime().register_library(description)
