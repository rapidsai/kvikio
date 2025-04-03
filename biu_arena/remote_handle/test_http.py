import cupy as cp
import kvikio
import kvikio.defaults
import time


def pread_one(path):
    print("read", path)
    rf = kvikio.RemoteFile.open_http(path)
    buf = cp.empty(rf.nbytes() // 8)
    return (rf.pread(buf), rf, buf)


def main():

    paths = [
        f"https://noaa-nwm-retrospective-3-0-pds.s3.amazonaws.com/CONUS/netcdf/FORCING/2023/20230101{i:0>2d}00.LDASIN_DOMAIN1" for i in range(1)]

    start = time.time()
    tuples = [pread_one(path) for path in paths]
    for i, (fut, _, _) in enumerate(tuples):
        print("get", i, time.monotonic())
        fut.get()

    elapsed = time.time() - start
    print(f"Elapsed time: {elapsed}")


if __name__ == "__main__":
    main()
