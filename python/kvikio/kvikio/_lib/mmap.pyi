# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.

import os
from typing import Optional, Union

from kvikio.cufile import IOFuture

class InternalMmapHandle:
    def __init__(
        self,
        file_path: os.PathLike[str],
        flags: str = "r",
        initial_map_size: Optional[int] = None,
        initial_map_offset: int = 0,
        mode: int = 420,  # stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IROTH
        map_flags: Optional[int] = None,
    ) -> None: ...
    def initial_map_size(self) -> int: ...
    def initial_map_offset(self) -> int: ...
    def file_size(self) -> int: ...
    def close(self) -> None: ...
    def closed(self) -> bool: ...
    def read(
        self,
        buf: Union[memoryview, bytearray, bytes],  # buffer-like
        size: Optional[int] = None,
        offset: int = 0,
    ) -> int: ...
    def pread(
        self,
        buf: Union[memoryview, bytearray, bytes],  # buffer-like
        size: Optional[int] = None,
        offset: int = 0,
        task_size: Optional[int] = None,
    ) -> IOFuture: ...
