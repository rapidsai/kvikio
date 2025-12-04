# SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from typing import Any, overload

import kvikio._lib.defaults
from kvikio.utils import call_once


class ConfigContextManager:
    """Context manager allowing the KvikIO configurations to be set upon entering a
    `with` block, and automatically reset upon leaving the block.
    """

    def __init__(self, config: dict[str, str]):
        (
            self._property_getters,
            self._property_setters,
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
        if property == "num_threads":
            property = "thread_pool_nthreads"
        func = self._property_getters[property]
        return func()

    def _set_property(self, property: str, value: Any):
        if property == "num_threads":
            property = "thread_pool_nthreads"
        func = self._property_setters[property]
        func(value)

    @call_once
    def _property_getter_and_setter(self) -> tuple[dict[str, Any], dict[str, Any]]:
        module_dict = vars(kvikio._lib.defaults)

        property_getter_names = [
            "compat_mode",
            "thread_pool_nthreads",
            "task_size",
            "gds_threshold",
            "bounce_buffer_size",
            "http_max_attempts",
            "http_status_codes",
            "http_timeout",
            "auto_direct_io_read",
            "auto_direct_io_write",
        ]

        property_getters = {}
        property_setters = {}

        for name in property_getter_names:
            property_getters[name] = module_dict[name]
            property_setters[name] = module_dict["set_" + name]
        return property_getters, property_setters


@overload
def set(config: dict[str, Any], /) -> ConfigContextManager: ...


@overload
def set(key: str, value: Any, /) -> ConfigContextManager: ...


def set(*config) -> ConfigContextManager:
    """Set KvikIO configurations.

    Examples:

    - To set one or more properties

      .. code-block:: python

         # Set the property globally.
         kvikio.defaults.set({"prop1": value1, "prop2": value2})

         # Set the property with a context manager.
         # The property automatically reverts to its old value
         # after leaving the `with` block.
         with kvikio.defaults.set({"prop1": value1, "prop2": value2}):
             ...

    - To set a single property

      .. code-block:: python

         # Set the property globally.
         kvikio.defaults.set("prop", value)

         # Set the property with a context manager.
         # The property automatically reverts to its old value
         # after leaving the `with` block.
         with kvikio.defaults.set("prop", value):
             ...

    Parameters
    ----------
    config
        The configurations. Can either be a single parameter (dict) consisting of one
        or more properties, or two parameters key (string) and value (Any)
        indicating a single property.

        Valid configuration names are:

        - ``"compat_mode"``
        - ``"num_threads"``
        - ``"task_size"``
        - ``"gds_threshold"``
        - ``"bounce_buffer_size"``
        - ``"http_max_attempts"``
        - ``"http_status_codes"``
        - ``"http_timeout"``
        - ``"auto_direct_io_read"``
        - ``"auto_direct_io_write"``

    Returns
    -------
    ConfigContextManager
       A context manager. If used in a `with` statement, the configuration will revert
       to its old value upon leaving the block.
    """

    err_msg = (
        "Valid arguments are kvikio.defaults.set(config: dict) or "
        "kvikio.defaults.set(key: str, value: Any)"
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
    """Get KvikIO configurations.

    Parameters
    ----------
    config_name: str
        The name of the configuration.

        Valid configuration names are:

        - ``"compat_mode"``
        - ``"num_threads"``
        - ``"task_size"``
        - ``"gds_threshold"``
        - ``"bounce_buffer_size"``
        - ``"http_max_attempts"``
        - ``"http_status_codes"``
        - ``"http_timeout"``

    Returns
    -------
    Any
        The value of the configuration.
    """
    context_manager = ConfigContextManager({})
    return context_manager._get_property(config_name)


def is_compat_mode_preferred() -> bool:
    return kvikio._lib.defaults.is_compat_mode_preferred()
