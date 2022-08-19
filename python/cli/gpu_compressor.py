# NVIDIA 2022

import argparse
import os
import sys
import time

import cupy

import kvikio
import kvikio.nvcomp as nvcomp

if __name__ == "__main__":

    class NvcompParser(argparse.ArgumentParser):
        """
        Handle special case and show help on invalid argument
        """

        def error(self, message):
            sys.stderr.write("\nERROR: {}\n\n".format(message))
            self.print_help()
            sys.exit(2)

    parser = NvcompParser()
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose Output")
    parser.add_argument(
        "-o",
        "--out_file",
        action="store",
        dest="out_file",
        help="Output filename",
    )
    parser.add_argument(
        "-c",
        choices=["ans", "bitcomp", "cascaded", "gdeflate", "lz4", "snappy"],
        action="store",
        dest="compression",
        help="Which GPU algorithm to use for compression.",
    )
    parser.add_argument(
        "-d",
        action="store_true",
        help="Decompress the incoming file",
    )
    parser.add_argument(action="store", dest="filename", help="Relative Filename")
    args = parser.parse_args()

    print("GPU Compression Initialized") if args.verbose else None

    file_size = os.path.getsize(args.filename)
    """ test
    data = cupy.arange(10000, dtype="uint8")
    """
    data = cupy.zeros(file_size, dtype=cupy.int8)
    t = time.time()
    f = kvikio.CuFile(args.filename, "r")
    f.read(data)
    f.close()
    read_time = time.time() - t
    print(f"File read time: {read_time:.3} seconds.") if args.verbose else None

    if args.d:
        compressor = nvcomp.ManagedDecompressionManager(data)
    elif args.compression == "ans":
        compressor = nvcomp.ANSManager()
    elif args.compression == "bitcomp":
        compressor = nvcomp.BitcompManager()
    elif args.compression == "cascaded":
        compressor = nvcomp.CascadedManager()
    elif args.compression == "gdeflate":
        compressor = nvcomp.GdeflateManager()
    elif args.compression == "snappy":
        compressor = nvcomp.SnappyManager()
    else:
        compressor = nvcomp.LZ4Manager(chunk_size=1 << 16)

    if args.d is True:
        print(f"Decompressing {file_size} bytes") if args.verbose else None
        t = time.time()
        converted = compressor.decompress(data)
        decompress_time = time.time() - t
        print(
            f"Decompression time: {decompress_time:.3} seconds"
        ) if args.verbose else None

        if not args.out_file:
            raise ValueError("Must specify filename with -o for decompression.")

        t = time.time()
        o = kvikio.CuFile(args.out_file, "w")
        o.write(converted)
        o.close()
        io_time = time.time() - t
        print(f"File write time: {io_time:.3} seconds") if args.verbose else None

        print(
            f"Decompressed file size {os.path.getsize(args.out_file)}"
        ) if args.verbose else None
    else:
        file_size = os.path.getsize(args.filename)

        print(f"Compressing {file_size} bytes") if args.verbose else None
        t = time.time()
        converted = compressor.compress(data)
        compress_time = time.time() - t
        print(f"Compression time: {compress_time:.3} seconds") if args.verbose else None

        t = time.time()
        if args.out_file:
            o = kvikio.CuFile(args.out_file, "w")
        else:
            o = kvikio.CuFile(args.filename + ".gpc", "w")
        o.write(converted)
        o.close()
        io_time = time.time() - t
        print(f"File write time: {io_time:.3} seconds") if args.verbose else None

        print(
            f"Compressed file size {compressor.get_compressed_output_size(converted)}"
        ) if args.verbose else None

    if args.out_file:
        end_name = args.out_file
    else:
        end_name = args.filename + ".gpc"
    print(f"Created file {end_name}") if args.verbose else None
