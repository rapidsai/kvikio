import cupy as cp
import kvikio
import kvikio.defaults
import time


def pread_one(path):
    print("read", path)
    rf = kvikio.RemoteFile.open_s3("kvikiobench-33622", path)
    buf = cp.empty(rf.nbytes() // 8)
    return (rf.pread(buf), rf, buf)


def main():
    kvikio.defaults.num_threads_reset(8)

    paths = [
        f"data/parquet-many/timeseries.parquet/part.{i}.parquet" for i in range(10)
    ]

    start = time.time()
    tuples = [pread_one(path) for path in paths]
    for i, (fut, _, _) in enumerate(tuples):
        print("get", i, time.monotonic())
        fut.get()

    elapsed = time.time() - start
    print(f"Elapsed time: {elapsed}")


if __name__ == "__main__":
    main()
