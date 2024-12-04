# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.

import pytest

import kvikio.cufile_driver


def test_version():
    major, minor = kvikio.cufile_driver.libcufile_version()
    assert major >= 0
    assert minor >= 0


@pytest.mark.cufile
def test_open_and_close():
    kvikio.cufile_driver.driver_open()
    kvikio.cufile_driver.driver_close()
