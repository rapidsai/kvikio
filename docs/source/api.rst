API
===

CuFile
------
.. currentmodule:: kvikio.cufile

.. autoclass:: CuFile
    :members:

.. autoclass:: IOFuture
    :members:

.. autofunction:: get_page_cache_info

.. autofunction:: clear_page_cache

CuFile driver
-------------
.. currentmodule:: kvikio.cufile_driver

.. autoclass:: ConfigContextManager

.. autofunction:: set

.. autofunction:: get

.. autofunction:: libcufile_version

.. autofunction:: driver_open

.. autofunction:: driver_close

.. autofunction:: initialize

Mmap
----
.. currentmodule:: kvikio.mmap

.. autoclass:: Mmap
    :members:

Zarr
----
.. currentmodule:: kvikio.zarr

.. autoclass:: GDSStore
    :members:

RemoteFile
----------
.. currentmodule:: kvikio.remote_file

.. autoclass:: RemoteFile
    :members:

Defaults
--------
.. currentmodule:: kvikio.defaults

.. autoclass:: ConfigContextManager

.. autofunction:: set

.. autofunction:: get

.. autofunction:: compat_mode

.. autofunction:: num_threads

.. autofunction:: task_size

.. autofunction:: gds_threshold

.. autofunction:: bounce_buffer_size

.. autofunction:: http_status_codes

.. autofunction:: http_max_attempts

.. autofunction:: compat_mode_reset
.. autofunction:: set_compat_mode

.. autofunction:: num_threads_reset
.. autofunction:: set_num_threads

.. autofunction:: task_size_reset
.. autofunction:: set_task_size

.. autofunction:: gds_threshold_reset
.. autofunction:: set_gds_threshold

.. autofunction:: bounce_buffer_size_reset
.. autofunction:: set_bounce_buffer_size

.. autofunction:: http_status_codes_reset
.. autofunction:: set_http_status_codes

.. autofunction:: http_max_attempts_reset
.. autofunction:: set_http_max_attempts
