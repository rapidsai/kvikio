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

.. autoclass:: RemoteEndpointType

.. autoclass:: RemoteFile
    :members:

Defaults
--------
.. currentmodule:: kvikio.defaults

.. autoclass:: ConfigContextManager

.. autofunction:: set

.. autofunction:: get
