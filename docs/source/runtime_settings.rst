Runtime Settings
================

Compatibility Mode ``KVIKIO_COMPAT_MODE``
-----------------------------------------
When KvikIO is running in compatibility mode, it doesn't load ``libcufile.so``. Instead, reads and writes are done using POSIX. Notice, this is not the same as the compatibility mode in cuFile. It is possible that KvikIO performs I/O in the non-compatibility mode by using the cuFile library, but the cuFile library itself is configured to operate in its own compatibility mode. For more details, refer to `cuFile compatibility mode <https://docs.nvidia.com/gpudirect-storage/api-reference-guide/index.html#cufile-compatibility-mode>`_ and `cuFile environment variables <https://docs.nvidia.com/gpudirect-storage/troubleshooting-guide/index.html#environment-variables>`_ .

The environment variable ``KVIKIO_COMPAT_MODE`` has three options (case-insensitive):

  * ``ON`` (aliases: ``TRUE``, ``YES``, ``1``): Enable the compatibility mode.
  * ``OFF`` (aliases: ``FALSE``, ``NO``, ``0``): Disable the compatibility mode, and enforce cuFile I/O. GDS will be activated if the system requirements for cuFile are met and cuFile is properly configured. However, if the system is not suited for cuFile, I/O operations under the ``OFF`` option may error out.
  * ``AUTO``: Try cuFile I/O first, and fall back to POSIX I/O if the system requirements for cuFile are not met.

Under ``AUTO``, KvikIO falls back to the compatibility mode:

  * when ``libcufile.so`` cannot be found.
  * when running in Windows Subsystem for Linux (WSL).
  * when ``/run/udev`` isn't readable, which typically happens when running inside a docker image not launched with ``--volume /run/udev:/run/udev:ro``.

This setting can be queried (:py:func:`kvikio.defaults.get`) and modified (:py:func:`kvikio.defaults.set`) at runtime using the property name ``compat_mode``.

Thread Pool ``KVIKIO_NTHREADS``
-------------------------------
KvikIO can use multiple threads for IO automatically. Set the environment variable ``KVIKIO_NTHREADS`` to the number of threads in the thread pool. If not set, the default value is 1.

This setting can be queried (:py:func:`kvikio.defaults.get`) and modified (:py:func:`kvikio.defaults.set`) at runtime using the property name ``num_threads``.

Task Size ``KVIKIO_TASK_SIZE``
------------------------------
KvikIO splits parallel IO operations into multiple tasks. Set the environment variable ``KVIKIO_TASK_SIZE`` to the maximum task size (in bytes). If not set, the default value is 4194304 (4 MiB).

This setting can be queried (:py:func:`kvikio.defaults.get`) and modified (:py:func:`kvikio.defaults.set`) at runtime using the property name ``task_size``.

GDS Threshold ``KVIKIO_GDS_THRESHOLD``
--------------------------------------
In order to improve performance of small IO, ``.pread()`` and ``.pwrite()`` implement a shortcut that circumvent the threadpool and use the POSIX backend directly. Set the environment variable ``KVIKIO_GDS_THRESHOLD`` to the minimum size (in bytes) to use GDS. If not set, the default value is 16384 (16 KiB).

This setting can be queried (:py:func:`kvikio.defaults.get`) and modified (:py:func:`kvikio.defaults.set`) at runtime using the property name ``gds_threshold``.

Size of the Bounce Buffer ``KVIKIO_BOUNCE_BUFFER_SIZE``
-------------------------------------------------------
KvikIO might have to use intermediate host buffers (one per thread) when copying between files and device memory. Set the environment variable ``KVIKIO_BOUNCE_BUFFER_SIZE`` to the size (in bytes) of these "bounce" buffers. If not set, the default value is 16777216 (16 MiB).

This setting can be queried (:py:func:`kvikio.defaults.get`) and modified (:py:func:`kvikio.defaults.set`) at runtime using the property name ``bounce_buffer_size``.

HTTP Retries ``KVIKIO_HTTP_STATUS_CODES``, ``KVIKIO_HTTP_MAX_ATTEMPTS``
------------------------------------------------------------------------

The behavior when a remote I/O read returns an error can be controlled through the `KVIKIO_HTTP_STATUS_CODES`, `KVIKIO_HTTP_MAX_ATTEMPTS`, and `KVIKIO_HTTP_TIMEOUT` environment variables.

KvikIO will retry a request should any of the HTTP status code in ``KVIKIO_HTTP_STATUS_CODES`` is received. The default values are ``429, 500, 502, 503, 504``. This setting can be queried (:py:func:`kvikio.defaults.get`) and modified (:py:func:`kvikio.defaults.set`) at runtime using the property name ``http_status_codes``.

The maximum number of attempts to make before throwing an exception is controlled by ``KVIKIO_HTTP_MAX_ATTEMPTS``. The default value is 3. This setting can be queried (:py:func:`kvikio.defaults.get`) and modified (:py:func:`kvikio.defaults.set`) at runtime using the property name ``http_max_attempts``.

The maximum duration of each HTTP request is controlled by ``KVIKIO_HTTP_TIMEOUT``. The default value is 60, which is the duration in seconds to allow. This setting can be queried (:py:func:`kvikio.defaults.get`) and modified (:py:func:`kvikio.defaults.set`) at runtime using the property name ``http_timeout``.

HTTP Verbose ``KVIKIO_REMOTE_VERBOSE``
--------------------------------------

For debugging HTTP requests, you can enable verbose output that shows detailed information about HTTP communication including headers, request/response bodies, connection details, and SSL handshake information.

Set the environment variable ``KVIKIO_REMOTE_VERBOSE`` to ``true``, ``on``, ``yes``, or ``1`` (case-insensitive) to enable verbose output. Otherwise, verbose output is disabled by default.

.. warning::

   This may show sensitive contents from headers and data.

CA bundle file and CA directory ``CURL_CA_BUNDLE``, ``SSL_CERT_FILE``, ``SSL_CERT_DIR``
---------------------------------------------------------------------------------------

The Certificate Authority (CA) paths required for TLS/SSL verification in ``libcurl`` can be explicitly specified using the following environment variables in order of overriding priority:

  * ``CURL_CA_BUNDLE`` (also used in the ``curl`` program) or ``SSL_CERT_FILE`` (also used in OpenSSL): Specifies the CA certificate bundle file location.
  * ``SSL_CERT_DIR`` (also used in OpenSSL): Specifies the CA certificate directory.

When neither is specified, KvikIO searches several standard system locations for the CA file and directory, and if the search fails falls back to the libcurl compile-time defaults.

Opportunistic POSIX Direct I/O operations ``KVIKIO_AUTO_DIRECT_IO_READ``, ``KVIKIO_AUTO_DIRECT_IO_WRITE``
---------------------------------------------------------------------------------------------------------

Overview
^^^^^^^^

By default, POSIX I/O operations perform buffered I/O using the OS page cache. However, Direct I/O (bypassing the page cache) can significantly improve performance in certain scenarios, such as writes and cold page-cache reads.

Traditional Direct I/O has strict requirements: The buffer address must be page-aligned, the file offset must be page-aligned, and the transfer size must be a multiple of page size (typically 4096 bytes). :py:class:`kvikio.CuFile` provides the feature of opportunistic Direct I/O, which removes these restrictions by automatically handling alignment. Specifically, KvikIO can split a POSIX I/O operation into unaligned and aligned segments and apply buffered I/O and direct I/O respectively.

Configuration
^^^^^^^^^^^^^

Set the environment variable ``KVIKIO_AUTO_DIRECT_IO_READ`` / ``KVIKIO_AUTO_DIRECT_IO_WRITE`` to ``true``, ``on``, ``yes``, or ``1`` (case-insensitive) to enable opportunistic Direct I/O.

.. code-block:: bash

   export KVIKIO_AUTO_DIRECT_IO_READ=1
   export KVIKIO_AUTO_DIRECT_IO_WRITE=1

Set them to ``false``, ``off``, ``no``, or ``0`` to disable this feature and use buffered I/O.

.. code-block:: bash

   export KVIKIO_AUTO_DIRECT_IO_READ=0
   export KVIKIO_AUTO_DIRECT_IO_WRITE=0

If not set, the default setting is buffered I/O for POSIX read (``KVIKIO_AUTO_DIRECT_IO_READ=0``) and Direct I/O for POSIX write (``KVIKIO_AUTO_DIRECT_IO_WRITE=1``).

Programmatic Access
^^^^^^^^^^^^^^^^^^^

These settings can be queried (:py:func:`kvikio.defaults.get`) and modified (:py:func:`kvikio.defaults.set`) at runtime using the property name ``auto_direct_io_read`` and ``auto_direct_io_write``.

Example:

.. code-block:: python

   import kvikio.defaults

   # Check current settings
   print(kvikio.defaults.get("auto_direct_io_read"))
   print(kvikio.defaults.get("auto_direct_io_write"))

   # Enable Direct I/O for reads, and disable it for writes
   kvikio.defaults.set({"auto_direct_io_read": True, "auto_direct_io_write": False})

Remote Backend ``KVIKIO_REMOTE_BACKEND``
----------------------------------------

KvikIO supports multiple libcurl-based backends for fetching data from remote endpoints (S3, HTTP, etc.). Set the environment variable ``KVIKIO_REMOTE_BACKEND`` to one of the following options (case-insensitive):

  * ``LIBCURL_EASY``: Use libcurl's easy interface with a thread pool for parallelism. Each chunk is fetched by a separate thread using blocking ``curl_easy_perform()`` calls. This is the default backend.
  * ``LIBCURL_MULTI_POLL``: Use libcurl's multi interface with poll-based concurrent transfers. A single call manages multiple concurrent connections using ``curl_multi_poll()``, with k-way buffering to overlap network I/O with host-to-device transfers.

If not set, the default value is ``LIBCURL_EASY``.

.. note::

   Changing this setting after creating a ``RemoteHandle`` has no effect on existing handles. The backend is determined at ``RemoteHandle`` construction time.

This setting can be queried (:py:func:`kvikio.defaults.get`) and modified (:py:func:`kvikio.defaults.set`) at runtime using the property name ``remote_backend``.

Remote Max Connections ``KVIKIO_REMOTE_MAX_CONNECTIONS``
--------------------------------------------------------

When using the ``LIBCURL_MULTI_POLL`` backend, this setting controls the maximum number of concurrent HTTP connections used for parallel chunk downloads. Set the environment variable ``KVIKIO_REMOTE_MAX_CONNECTIONS`` to a positive integer.

If not set, the default value is 8.

.. note::

   This setting only applies when using ``RemoteBackendType.LIBCURL_MULTI_POLL``. It has no effect on the ``LIBCURL_EASY`` backend, which uses the thread pool size (``KVIKIO_NTHREADS``) to control parallelism.

This setting can be queried (:py:func:`kvikio.defaults.get`) and modified (:py:func:`kvikio.defaults.set`) at runtime using the property name ``remote_max_connections``.

Number of Bounce Buffers ``KVIKIO_NUM_BOUNCE_BUFFERS``
------------------------------------------------------

When using the ``LIBCURL_MULTI_POLL`` backend with device memory destinations, KvikIO uses k-way buffering to overlap network I/O with host-to-device memory transfers. This setting controls the number of bounce buffers allocated per connection.

Set the environment variable ``KVIKIO_NUM_BOUNCE_BUFFERS`` to a positive integer. Higher values allow more overlap between network I/O and H2D transfers but consume more pinned memory. The total pinned memory usage is ``remote_max_connections * num_bounce_buffers * bounce_buffer_size``.

If not set, the default value is 2.

.. note::

   This setting only applies when using ``RemoteBackendType.LIBCURL_MULTI_POLL`` with device memory destinations. For host memory destinations or the ``LIBCURL_EASY`` backend, bounce buffers are not used.

This setting can be queried (:py:func:`kvikio.defaults.get`) and modified (:py:func:`kvikio.defaults.set`) at runtime using the property name ``num_bounce_buffers``.
