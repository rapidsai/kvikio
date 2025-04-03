import kvikio
import kvikio.defaults
import kvikio.cufile_driver
import subprocess
import os.path
import cupy
import numpy as np
import time


class TestManager:
    def __init__(self):
        self.arena_dir = "/mnt/nvme"
        self.filename = os.path.join(self.arena_dir, "parallel_io.bin")
        self.Mi = 1024 * 1024
        self.num_elements = 4 * self.Mi / 8
        self.num_threads = 2
        self.task_size = 4 * 1024 * 1024

        self.host_data = np.arange(0, self.num_elements, dtype=np.float64)

        kvikio.defaults.set("num_threads", 8)

        print("--> version: {:}".format(kvikio.__version__))
        print("    num_threads: {:}".format(
            kvikio.defaults.get("num_threads")))
        print("    task_size: {:}".format(
            kvikio.defaults.get("task_size")))
        print("    gds_threshold: {:}".format(
            kvikio.defaults.get("gds_threshold")))
        print("    gds available:  {:}".format(
            kvikio.cufile_driver.get("is_gds_available")))
        print("    gds available:  {:}".format(
            kvikio.cufile_driver.properties.is_gds_available))

        kvikio.defaults.set(
            {"compat_mode": True,
             "num_threads": self.num_threads,
             "task_size": self.task_size})

    def _drop_file_cache(self):
        full_command = "sudo /sbin/sysctl vm.drop_caches=3"
        subprocess.run(full_command.split())

    def write_to_file(self):
        print("--> Write to file")
        self.host_data.tofile(self.filename)

    def read_from_file(self):
        print("--> Read from file to device memory")

        print("    num_threads: {:}".format(
            kvikio.defaults.get("num_threads")))
        with kvikio.defaults.set("num_threads", 8), kvikio.CuFile(self.filename, "r") as file_handle:
            dev_buf = cupy.empty_like(self.host_data)
            fut = file_handle.pread(dev_buf)
            bytes_read = fut.get()
            print("    bytes_read: {:}".format(bytes_read))
            print("    num_threads: {:}".format(
                kvikio.defaults.get("num_threads")))
        print("    num_threads: {:}".format(
            kvikio.defaults.get("num_threads")))


if __name__ == "__main__":
    tm = TestManager()

    tm.write_to_file()
    tm.read_from_file()
