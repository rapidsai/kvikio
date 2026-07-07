Profiling
=========

NIC bandwidth on the Nsight Systems timeline
--------------------------------------------
KvikIO ships a standalone `Nsight Systems <https://developer.nvidia.com/nsight-systems>`_ (nsys) plugin named ``kvikio_nic`` that samples per-interface network bandwidth and places it on the nsys timeline.

The plugin reads the Linux kernel byte counters (``/sys/class/net/<interface>/statistics/{rx,tx}_bytes``) at a fixed interval and emits one NVTX counter group per monitored interface plus a ``total`` group, all under the ``KvikIO NIC`` NVTX domain. Each group carries the receive (``rx``) and transmit (``tx``) rates.

Installation
------------
The plugin ships by default on both packaging channels, inside the ``libkvikio`` conda package and inside the ``libkvikio`` wheel. It consists of an executable (``kvikio_nic_nsys_plugin``) and a manifest (``nsys-plugin.yaml``) placed together in a ``kvikio_nic`` directory. Use :py:func:`kvikio.nsys_plugin_search_dir` to locate that directory on either channel.

Command line options
--------------------
  * ``-i | --interval``: Sampling interval in microseconds (default 20000, i.e. 50 Hz).
  * ``-d | --device``: Interface name regex. If not given, all up (or unknown) non-loopback interfaces are monitored. An explicit regex selects interfaces by name and bypasses the up check, so an explicit request is always honored.

On the nsys command line, plugin arguments follow the plugin name as a comma separated list. These are equivalent:

.. code-block:: bash

   nsys profile --enable=kvikio_nic,-d,eth0,-i,20000 ...
   nsys profile --enable=kvikio_nic,-deth0,-i20000 ...
   nsys profile --enable=kvikio_nic,--device=eth0,--interval=20000 ...

Usage with nsys 2026.2.1 or newer
---------------------------------
Nsight Systems 2026.2.1 introduced ``NSYS_PLUGIN_SEARCH_DIRS`` for discovering third party plugins:

.. code-block:: bash

   export NSYS_PLUGIN_SEARCH_DIRS="$(python -c 'import kvikio; print(kvikio.nsys_plugin_search_dir())')"
   nsys profile --enable=help # should list kvikio_nic
   nsys profile --enable=kvikio_nic,--device=eth0 --trace=cuda,nvtx,osrt -o nsys_report my_application

Usage with older nsys
---------------------
Older nsys versions only discover plugins inside the nsys installation itself. Either copy or symlink the ``kvikio_nic`` directory next to the bundled plugins (write access to the installation is needed, typically root):

.. code-block:: bash

   sudo ln -s "$(python -c 'import kvikio; print(kvikio.nsys_plugin_search_dir())')/kvikio_nic" /opt/nvidia/nsight-systems/<version>/target-linux-x64/plugins/
