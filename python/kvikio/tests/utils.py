# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.

import socket


def localhost() -> str:
    return "127.0.0.1"


def find_free_port(host: str = localhost()) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, 0))
        _, port = s.getsockname()
    return port
