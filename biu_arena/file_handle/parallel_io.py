import kvikio
import kvikio.defaults
import subprocess
import os.path
import cupy
import numpy as np
import time
import nvtx


class TestManager:
    def __init__(self, config: dict):
        self.arena_dir = "/mnt/nvme"
        self.filename = os.path.join(self.arena_dir, "parallel_io.bin")
        self.Mi = 1024 * 1024

        self.config = {}
        self.config["num_elements"] = config.get(
            "num_elements", 32 * 1024 * self.Mi / 8)
        self.config["num_threads"] = config.get("num_threads", 72)
        self.config["task_size"] = config.get("task_size", 4 * 1024 * 1024)
        self.config["repetition"] = config.get("repetition", 11)
        self.config["compat_mode"] = config.get("compat_mode", True)
        self.config["create_file"] = config.get("create_file", True)
        self.config["skip_remaining_tests"] = config.get(
            "skip_remaining_tests", False)
        self.config["drop_file_cache"] = config.get("drop_file_cache", True)

        self.num_elements = self.config["num_elements"]
        self.num_threads = self.config["num_threads"]
        self.task_size = self.config["task_size"]
        self.create_file = self.config["create_file"]
        self.skip_remaining_tests = self.config["skip_remaining_tests"]
        self.drop_file_cache = self.config["drop_file_cache"]

        self.host_data = np.arange(0, self.num_elements, dtype=np.float64)
        self.repetition = self.config["repetition"]
        self.compat_mode = self.config["compat_mode"]

        kvikio.defaults.set(
            {"compat_mode": self.compat_mode,
             "num_threads": self.num_threads,
             "task_size": self.task_size})

        print("--> parameter: {:}".format(str(self.config)))

    def _drop_file_cache(self):
        full_command = "sudo /sbin/sysctl vm.drop_caches=3"
        subprocess.run(full_command.split())

    def write_to_file(self):
        with nvtx.annotate("write_to_file"):
            print("--> Write to file")
            self.host_data.tofile(self.filename)

    def read_from_file(self):
        with nvtx.annotate("read_from_file"):
            print("--> Read from file to device memory")
            file_handle = kvikio.CuFile(self.filename, "r")
            dev_buf = cupy.empty_like(self.host_data)
            total_time = 0
            count = 0

            for idx in range(self.repetition):
                if self.drop_file_cache:
                    self._drop_file_cache()

                start = time.time()
                fut = file_handle.pread(dev_buf)
                fut.get()
                elapsed_time = time.time() - start

                if idx > 0:
                    total_time += elapsed_time
                    count += 1

                bandwidth = self.num_elements * 8 / elapsed_time / self.Mi
                print(
                    "    {:4d} - {:.6f} [s], {:.6f} [MiB/s]".format(idx, elapsed_time, bandwidth))

            average_bandwidth = self.num_elements * 8 * \
                count / total_time / self.Mi
            print(
                "    Total time: {:.4f} [s], average bandwidth: {:.4f} [MiB/s]".format(total_time, average_bandwidth))

    def perform_test(self):
        if self.create_file:
            self.write_to_file()

        if self.skip_remaining_tests:
            return

        self.read_from_file()


if __name__ == "__main__":
    config = {}
    tm = TestManager(config)
    tm.perform_test()
