# https://github.com/rapidsai/kvikio/issues/629


import kvikio.defaults
import kvikio
import os.path
import time
import nvtx
import numpy as np
import cupyx
import cupy
import subprocess


class TestManager:
    def __init__(self, config: dict):
        self.arena_dir = "/mnt/nvme"
        self.filename = os.path.join(self.arena_dir, "file_vs_host.bin")
        self.Mi = 1024 * 1024

        self.config = {}
        self.config["num_elements"] = config.get(
            "num_elements", 8 * 1024 * self.Mi / 8)
        self.config["num_threads"] = config.get("num_threads", 72)
        self.config["task_size"] = config.get("task_size", 4 * 1024 * 1024)
        self.config["repetition"] = config.get("repetition", 11)
        self.config["compat_mode"] = config.get("compat_mode", True)
        self.config["create_file"] = config.get("create_file", True)
        self.config["skip_remaining_tests"] = config.get(
            "skip_remaining_tests", False)
        self.config["skip_mmap"] = config.get("skip_mmap", False)
        self.config["drop_file_cache"] = config.get("drop_file_cache", True)

        self.num_elements = self.config["num_elements"]
        self.num_threads = self.config["num_threads"]
        self.task_size = self.config["task_size"]
        self.create_file = self.config["create_file"]
        self.skip_remaining_tests = self.config["skip_remaining_tests"]
        self.skip_mmap = self.config["skip_mmap"]
        self.drop_file_cache = self.config["drop_file_cache"]

        self.host_data = np.arange(0, self.num_elements, dtype=np.float64)

        self.repetition = self.config["repetition"]
        self.compat_mode = self.config["compat_mode"]

        kvikio.defaults.set(
            {"compat_mode": True,
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

    def write_to_host(self, pinned=True):
        with nvtx.annotate("write_to_host"):
            print(
                "--> Write to host ({:})".format("pinned" if pinned else "pageable"))
            host_buf = None
            if pinned:
                host_buf = cupyx.empty_like_pinned(self.host_data)
            else:
                host_buf = np.empty_like(self.host_data)

            cupy.cuda.runtime.memcpy(host_buf.ctypes.data,
                                     self.host_data.ctypes.data,
                                     self.host_data.nbytes,
                                     cupy.cuda.runtime.memcpyHostToHost)
            return host_buf

    def read_from_file(self):
        with nvtx.annotate("read_from_file"):
            print("--> Read from file to device memory")
            file_handle = kvikio.CuFile(self.filename, "r")
            dev_buf = cupy.empty_like(self.host_data)

            def func():
                fut = file_handle.pread(dev_buf)
                fut.get()

            def init_func():
                if self.drop_file_cache:
                    self._drop_file_cache()

            self.bench_func(func, "Python read file", init_func)

    def read_from_mmap(self):
        with nvtx.annotate("read_from_mmap"):
            import sys
            sys.path.append("/home/coder/kvikio/biu_arena/biu_io")
            import biu_mmap
            print("--> Read from mmap to device memory")
            dev_buf = cupy.empty_like(self.host_data)

            def func():
                biu_mmap.read(self.filename, dev_buf)

            self.bench_func(func, "Python read file")

    def read_from_host(self, host_buf, pinned=True):
        host_mem_type = "pinned" if pinned else "pageable"
        with nvtx.annotate("read_from_host ({:})".format(host_mem_type)):
            print(
                "--> Read from host ({:}) to device memory".format(host_mem_type))
            dev_buf = cupy.empty_like(self.host_data)
            stream = cupy.cuda.Stream(non_blocking=True)

            def func():
                cupy.cuda.runtime.memcpyAsync(dev_buf.data.ptr,
                                              host_buf.ctypes.data,
                                              host_buf.nbytes,
                                              cupy.cuda.runtime.memcpyHostToDevice,
                                              stream.ptr)
                cupy.cuda.runtime.streamSynchronize(stream.ptr)

            self.bench_func(func, "Python read host")

    def bench_func(self, func, nvtx_msg, init_func=None):
        total_time = 0
        count = 0

        for idx in range(self.repetition):
            with nvtx.annotate(nvtx_msg):
                if init_func:
                    init_func()

                start = time.time()
                func()
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

        if not self.skip_mmap:
            self.read_from_mmap()

        host_buf = self.write_to_host()
        self.read_from_host(host_buf)

        host_buf = self.write_to_host(False)
        self.read_from_host(host_buf, False)


if __name__ == "__main__":
    config = {}
    tm = TestManager(config)
    tm.perform_test()
