# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.

import pytest

import kvikio.cufile_driver
import kvikio.defaults


@pytest.mark.skipif(
    kvikio.defaults.compat_mode(),
    reason=(
        "cannot test the cuFile driver when the "
        "test is running in compatibility mode"
    ),
)
def test_open_and_close():
    kvikio.cufile_driver.driver_open()
    kvikio.cufile_driver.driver_close()
