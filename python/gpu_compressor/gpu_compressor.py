# NVIDIA 2022

import argparse
import os
import sys
import time

import kvikio
import kvikio.nvcomp as nvcomp

import cupy

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
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Verbose Output"
    )
    parser.add_argument(
        "-o",
        "--out_file",
        action="store",
        dest="out_file",
        help="Output filename",
    )
    parser.add_argument(
        "-c",
        choices=["lz4", "cascaded", "snappy"],
        action="store",
        dest="compression",
        help="Which GPU algorithm to use for compression.",
    )
    parser.add_argument(
        "-d",
        action="store_true",
        help="Decompress the incoming file",
    )
    parser.add_argument(
        action="store", dest="filename", help="Relative Filename"
    )
    args = parser.parse_args()

    print("GPU Compression Initialized") if args.verbose else None

    file_size = os.path.getsize(args.filename)
    """ test
    data = cupy.arange(10000, dtype="uint8")
    """
    data = cupy.zeros(file_size, dtype=cupy.int8)
    f = kvikio.CuFile(args.filename, "r")
    f.read(data)
    f.close()

    if args.d:
        compressor = nvcomp.ManagedDecompressionManager(data)
    elif args.compression == "cascaded":
        compressor = nvcomp.CascadedManager()
    elif args.compression == "snappy":
        compressor = nvcomp.SnappyManager()
    else:
        compressor = nvcomp.LZ4Manager(chunk_size=1 << 16)

    if args.d == True:
        print(f"Decompressing {file_size} bytes") if args.verbose else None
        t = time.time()
        converted = compressor.decompress(data)
        if not args.out_file:
            raise ValueError(
                "Must specify filename with -o for decompression."
            )
        o = kvikio.CuFile(args.out_file, "w")
        o.write(converted)
        o.close()

        print(
            f"Decompressed file size {os.path.getsize(args.out_file)}"
        ) if args.verbose else None
    else:
        file_size = os.path.getsize(args.filename)
        print(f"Compressing {file_size} bytes") if args.verbose else None
        converted = compressor.compress(data)
        if args.out_file:
            o = kvikio.CuFile(args.out_file, "w")
        else:
            o = kvikio.CuFile(args.filename + ".gpc", "w")
        o.write(converted)
        o.close()
        print(
            f"Compressed file size {compressor.get_compressed_output_size(converted)}"
        ) if args.verbose else None
