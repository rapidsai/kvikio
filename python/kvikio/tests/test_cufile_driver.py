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


def test_property_setter():
    """Test the method `set`"""

    # Attempt to set a nonexistent property
    with pytest.raises(KeyError):
        kvikio.cufile_driver.set("nonexistent_property", 123)

    # Nested context managers
    poll_thresh_size_default = kvikio.cufile_driver.properties.poll_thresh_size
    with kvikio.cufile_driver.set("poll_thresh_size", 1024):
        assert kvikio.cufile_driver.properties.poll_thresh_size == 1024
        with kvikio.cufile_driver.set("poll_thresh_size", 2048):
            assert kvikio.cufile_driver.properties.poll_thresh_size == 2048
            with kvikio.cufile_driver.set("poll_thresh_size", 4096):
                assert kvikio.cufile_driver.properties.poll_thresh_size == 4096
            assert kvikio.cufile_driver.properties.poll_thresh_size == 2048
        assert kvikio.cufile_driver.properties.poll_thresh_size == 1024
    assert kvikio.cufile_driver.properties.poll_thresh_size == poll_thresh_size_default

    # Multiple context managers
    poll_mode_default = kvikio.cufile_driver.properties.poll_mode
    max_device_cache_size_default = kvikio.cufile_driver.properties.max_device_cache_size
    with kvikio.cufile_driver.set({"poll_mode": True, "max_device_cache_size": 2048}):
        assert kvikio.cufile_driver.properties.poll_mode and\
            (kvikio.cufile_driver.properties.max_device_cache_size == 2048)
    assert (kvikio.cufile_driver.properties.poll_mode == poll_mode_default) and\
        (kvikio.cufile_driver.properties.max_device_cache_size ==
         max_device_cache_size_default)
