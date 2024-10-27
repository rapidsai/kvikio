# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.

import kvikio.cufile_driver
import kvikio.defaults


def test_open_and_close():
    kvikio.cufile_driver.driver_open()
    kvikio.cufile_driver.driver_close()
