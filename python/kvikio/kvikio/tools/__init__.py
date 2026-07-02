# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""KvikIO tools: profiling and diagnostic utilities.

This subpackage holds standalone tools that help measure and profile KvikIO
I/O. The first tool is :mod:`kvikio.tools.remote_io_monitor`, which samples NIC
bandwidth and publishes it as an NVTX counter for the Nsight Systems timeline.
"""
