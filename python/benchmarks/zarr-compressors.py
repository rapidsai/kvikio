# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.

import argparse
import statistics
from dataclasses import dataclass
from enum import Enum
from time import perf_counter as clock

import cupy
import numcodecs.blosc
import numpy
from dask.utils import format_bytes, parse_bytes
from numcodecs.abc import Codec

import kvikio
import kvikio.defaults
import kvikio.zarr


class Device(Enum):
    CPU = 1
    GPU = 2


@dataclass
class Compressor:
    device: Device
    codec: Codec


compressors = {
    "lz4-default": Compressor(device=Device.CPU, codec=numcodecs.LZ4()),
    "lz4-blosc": Compressor(
        device=Device.CPU, codec=numcodecs.blosc.Blosc(cname="lz4")
    ),
    "lz4-nvcomp": Compressor(device=Device.GPU, codec=kvikio.zarr.LZ4()),
    "snappy-nvcomp": Compressor(device=Device.GPU, codec=kvikio.zarr.Snappy()),
    "cascaded-nvcomp": Compressor(device=Device.GPU, codec=kvikio.zarr.Cascaded()),
    "gdeflate-nvcomp": Compressor(device=Device.GPU, codec=kvikio.zarr.Gdeflate()),
    "bitcomp-nvcomp": Compressor(device=Device.GPU, codec=kvikio.zarr.Bitcomp()),
}


def create_src_data(args, compressor: Compressor):
    if compressor.device == Device.CPU:
        return numpy.random.random(args.nelem).astype(args.dtype)
    if compressor.device == Device.GPU:
        return cupy.random.random(args.nelem).astype(args.dtype)
    assert False, "Unknown device type"


def run(args, compressor: Compressor):
    src = create_src_data(args, compressor)
    dst = numpy.empty_like(src)  # Notice, if src is a cupy array dst is as well
    t0 = clock()
    a = compressor.codec.encode(src)
    encode_time = clock() - t0

    t0 = clock()
    compressor.codec.decode(a, out=dst)
    decode_time = clock() - t0
    return encode_time, decode_time


def main(args):
    cupy.cuda.set_allocator(None)  # Disable CuPy's default memory pool
    cupy.arange(10)  # Make sure CUDA is initialized

    try:
        import pynvml.smi

        nvsmi = pynvml.smi.nvidia_smi.getInstance()
    except ImportError:
        gpu_name = "Unknown (install pynvml)"
        mem_total = gpu_name
        bar1_total = gpu_name
    else:
        info = nvsmi.DeviceQuery()["gpu"][0]
        gpu_name = f"{info['product_name']} (dev #0)"
        mem_total = format_bytes(
            parse_bytes(
                str(info["fb_memory_usage"]["total"]) + info["fb_memory_usage"]["unit"]
            )
        )
        bar1_total = format_bytes(
            parse_bytes(
                str(info["bar1_memory_usage"]["total"])
                + info["bar1_memory_usage"]["unit"]
            )
        )

    nbytes = f"{format_bytes(args.nbytes)} bytes ({args.nbytes})"
    print("Encode/decode benchmark")
    print("----------------------------------")
    print(f"GPU                     | {gpu_name}")
    print(f"GPU Memory Total        | {mem_total}")
    print(f"BAR1 Memory Total       | {bar1_total}")
    print("----------------------------------")
    print(f"nbytes                  | {nbytes}")
    print(f"4K aligned              | {args.nbytes % 4096 == 0}")
    print(f"nruns                   | {args.nruns}")
    print("==================================")

    encode_output = ""
    decode_output = ""
    # Run each benchmark using the requested APIs
    for comp_name, comp in ((n, compressors[n]) for n in args.compressors):
        rs = []
        ws = []
        for _ in range(args.n_warmup_runs):
            encode, decode = run(args, comp)
        for _ in range(args.nruns):
            encode, decode = run(args, comp)
            rs.append(args.nbytes / encode)
            ws.append(args.nbytes / decode)

        def pprint_api_res(name, samples):
            mean = statistics.mean(samples) if len(samples) > 1 else samples[0]
            ret = f"{comp_name} {name}".ljust(24)
            ret += f"| {format_bytes(mean).rjust(10)}/s".ljust(14)
            if len(samples) > 1:
                stdev = statistics.stdev(samples) / mean * 100
                ret += " ± %5.2f %%" % stdev
                ret += " ("
                for sample in samples:
                    ret += f"{format_bytes(sample)}/s, "
                ret = ret[:-2] + ")"  # Replace trailing comma
            return ret

        encode_output += pprint_api_res("", rs) + "\n"
        decode_output += pprint_api_res("", ws) + "\n"
    print("Encode:")
    print(encode_output)
    print("Decode:")
    print(decode_output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Roundtrip benchmark")
    parser.add_argument(
        "-n",
        "--nbytes",
        metavar="BYTES",
        default="10 MiB",
        type=parse_bytes,
        help="Message size, which must be a multiple of 8 (default: %(default)s).",
    )
    parser.add_argument(
        "--dtype",
        default="float32",
        type=numpy.dtype,
        help="NumPy datatype to use (default: '%(default)s')",
    )
    parser.add_argument(
        "--nruns",
        metavar="RUNS",
        default=1,
        type=int,
        help="Number of runs per API (default: %(default)s).",
    )
    parser.add_argument(
        "--n-warmup-runs",
        default=0,
        type=int,
        help="Number of warmup runs (default: %(default)s).",
    )
    parser.add_argument(
        "--compressors",
        metavar="COMP_LIST",
        default="all",
        nargs="+",
        choices=tuple(compressors.keys()),
        help="List of compressors to use {%(choices)s} (default: all)",
    )

    args = parser.parse_args()
    if "all" in args.compressors:
        args.compressors = tuple(compressors.keys())

    # Check if size is divisible by size of datatype
    assert args.nbytes % args.dtype.itemsize == 0

    # Compute/convert to number of elements
    args.nelem = args.nbytes // args.dtype.itemsize

    main(args)
