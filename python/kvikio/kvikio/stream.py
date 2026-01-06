# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from kvikio._lib import stream as stream_module  # type: ignore


def stream_register(stream, flags: int) -> None:
    stream_module.stream_register(stream, flags)


def stream_deregister(stream) -> None:
    stream_module.stream_deregister(stream)
