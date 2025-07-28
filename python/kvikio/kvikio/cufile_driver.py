# Copyright (c) 2024-2025, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.

import atexit
from typing import Any, Tuple, overload

import kvikio.utils
from kvikio._lib import cufile_driver  # type: ignore

properties = cufile_driver.DriverProperties()
"""cuFile driver configurations. Use kvikio.cufile_driver.get and
   kvikio.cufile_driver.set to access the configurations.
"""


class ConfigContextManager:
    """Context manager allowing the cuFile driver configurations to be set upon
    entering a `with` block, and automatically reset upon leaving the block.
    """

    def __init__(self, config: dict[str, str]):
        (
            self._property_getters,
            self._property_setters,
            self._readonly_property_getters,
        ) = self._property_getter_and_setter()
        self._old_properties = {}

        for key, value in config.items():
            self._old_properties[key] = self._get_property(key)
            self._set_property(key, value)

    def __enter__(self):
        return None

    def __exit__(self, type_unused, value, traceback_unused):
        for key, value in self._old_properties.items():
            self._set_property(key, value)

    def _get_property(self, property: str) -> Any:
        if property in self._property_getters:
            func = self._property_getters[property]
        elif property in self._readonly_property_getters:
            func = self._readonly_property_getters[property]
        else:
            raise KeyError

        # getter signature: object.__get__(self, instance, owner=None)
        return func(properties)

    def _set_property(self, property: str, value: Any):
        if property in self._readonly_property_getters:
            raise KeyError("This property is read-only.")

        func = self._property_setters[property]

        # setter signature: object.__set__(self, instance, value)
        func(properties, value)

    @kvikio.utils.call_once
    def _property_getter_and_setter(
        self,
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        class_dict = vars(cufile_driver.DriverProperties)

        property_getter_names = [
            "poll_mode",
            "poll_thresh_size",
            "max_device_cache_size",
            "max_pinned_memory_size",
        ]

        property_getters = {}
        property_setters = {}

        for name in property_getter_names:
            property_getters[name] = class_dict[name].__get__
            property_setters[name] = class_dict[name].__set__

        readonly_property_getter_names = [
            "is_gds_available",
            "major_version",
            "minor_version",
            "allow_compat_mode",
            "per_buffer_cache_size",
        ]
        readonly_property_getters = {}
        for name in readonly_property_getter_names:
            readonly_property_getters[name] = class_dict[name].__get__

        return property_getters, property_setters, readonly_property_getters


@overload
def set(config: dict[str, Any], /) -> ConfigContextManager:
    ...


@overload
def set(key: str, value: Any, /) -> ConfigContextManager:
    ...


def set(*config) -> ConfigContextManager:
    """Set cuFile driver configurations.

    Examples:

    - To set one or more properties

      .. code-block:: python

         # Set the property globally.
         kvikio.cufile_driver.set({"prop1": value1, "prop2": value2})

         # Set the property with a context manager.
         # The property automatically reverts to its old value
         # after leaving the `with` block.
         with kvikio.cufile_driver.set({"prop1": value1, "prop2": value2}):
             ...

    - To set a single property

      .. code-block:: python

         # Set the property globally.
         kvikio.cufile_driver.set("prop", value)

         # Set the property with a context manager.
         # The property automatically reverts to its old value
         # after leaving the `with` block.
         with kvikio.cufile_driver.set("prop", value):
             ...

    Parameters
    ----------
    config
        The configurations. Can either be a single parameter (dict) consisting of one
        or more properties, or two parameters key (string) and value (Any)
        indicating a single property.

        Valid configuration names are:

        - Read-only properties:

          - ``"is_gds_available"``
          - ``"major_version"``
          - ``"minor_version"``
          - ``"allow_compat_mode"``
          - ``"per_buffer_cache_size"``

        - Settable properties:

          - ``"poll_mode"``
          - ``"poll_thresh_size"``
          - ``"max_device_cache_size"``
          - ``"max_pinned_memory_size"``

    Returns
    -------
    ConfigContextManager
       A context manager. If used in a `with` statement, the configuration will revert
       to its old value upon leaving the block.
    """

    err_msg = (
        "Valid arguments are kvikio.cufile_driver.set(config: dict) or "
        "kvikio.cufile_driver.set(key: str, value: Any)"
    )

    if len(config) == 1:
        if not isinstance(config[0], dict):
            raise ValueError(err_msg)
        return ConfigContextManager(config[0])
    elif len(config) == 2:
        if not isinstance(config[0], str):
            raise ValueError(err_msg)
        return ConfigContextManager({config[0]: config[1]})
    else:
        raise ValueError(err_msg)


def get(config_name: str) -> Any:
    """Get cuFile driver configurations.

    Parameters
    ----------
    config_name: str
        The name of the configuration.

        Valid configuration names are:

        - Read-only properties:

          - ``"is_gds_available"``
          - ``"major_version"``
          - ``"minor_version"``
          - ``"allow_compat_mode"``
          - ``"per_buffer_cache_size"``

        - Settable properties:

          - ``"poll_mode"``
          - ``"poll_thresh_size"``
          - ``"max_device_cache_size"``
          - ``"max_pinned_memory_size"``

    Returns
    -------
    Any
        The value of the configuration.
    """
    context_manager = ConfigContextManager({})
    return context_manager._get_property(config_name)


def libcufile_version() -> Tuple[int, int]:
    """Get the libcufile version.

    Returns (0, 0) for cuFile versions prior to v1.8.

    Notes
    -----
    This is not the version of the CUDA toolkit. cufile is part of the
    toolkit but follows its own version scheme.

    Returns
    -------
    The version as a tuple (MAJOR, MINOR).
    """
    v = cufile_driver.libcufile_version()
    # Convert the integer version like 1080 to (1, 8).
    major, minor = divmod(v, 1000)
    return (major, minor // 10)


def driver_open() -> None:
    """Open the cuFile driver

    cuFile accepts multiple calls to `driver_open()`. Only the first call
    opens the driver, but every call must have a matching call to
    `driver_close()`.

    Normally, it is not required to open and close the cuFile driver since
    it is done automatically.

    Raises
    ------
    RuntimeError
        If cuFile isn't available.
    """
    return cufile_driver.driver_open()


def driver_close() -> None:
    """Close the cuFile driver

    cuFile accepts multiple calls to `driver_open()`. Only the first call
    opens the driver, but every call must have a matching call to
    `driver_close()`.

    Raises
    ------
    RuntimeError
        If cuFile isn't available.
    """
    return cufile_driver.driver_close()


def initialize() -> None:
    """Open the cuFile driver and close it again at module exit

    Normally, it is not required to open and close the cuFile driver since
    it is done automatically.

    Notes
    -----
    Registers an atexit handler that calls :func:`driver_close`.

    Raises
    ------
    RuntimeError
        If cuFile isn't available.
    """
    driver_open()
    atexit.register(driver_close)
