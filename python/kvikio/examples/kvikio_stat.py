# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.

# usage: kvikio_stat [-h] --nsys-report-path NSYS_REPORT_PATH
#                    [--sql-path SQL_PATH]
#                    [--nsys-binary NSYS_BINARY]
#
# Generate I/O size histogram from Nsight System report.
#
# options:
#   -h, --help            show this help message and exit
#   --nsys-report-path NSYS_REPORT_PATH
#                         The path of the Nsight System report.
#   --sql-path SQL_PATH   kvikio_stat will invoke Nsight System to generated a SQL
#                         database from the provided Nsight System report. --sql-path
#                         specifies the path of this SQL database. If unspecified, the
#                         current working directory will be used to store it, and the
#                         file name will be derived from the Nsight System report.
#   --nsys-binary NSYS_BINARY
#                         The path of the Nsight System CLI program. If unspecified,
#                         "nsys" will be used.

import argparse
import os
import pathlib
import sqlite3
import subprocess

import numpy as np
import pandas as pd


class Analyzer:
    """_summary_"""

    def __init__(self, args: argparse.Namespace):
        """_summary_

        :param args: _description_
        :type args: argparse.Namespace
        """
        self.nsys_report_path = args.nsys_report_path

        if args.sql_path is None:
            report_basename_no_ext = pathlib.Path(self.nsys_report_path).stem
            self.sql_path = os.getcwd() + os.sep + report_basename_no_ext + ".sqlite"
        else:
            self.sql_path = args.sql_path

        if args.nsys_binary is None:
            self.nsys_binary = "nsys"
        else:
            self.nsys_binary = args.nsys_binary

    def _export_report_to_sqlite(self):
        """_summary_"""
        full_cmd_str = (
            f"{self.nsys_binary} export --type=sqlite --lazy=false "
            + f"--force-overwrite=true --output={self.sql_path} "
            + f"{self.nsys_report_path} "
            + "--tables=StringIds,NVTX_EVENTS"
        )
        full_cmd_list = full_cmd_str.split()
        print(f"Command: {full_cmd_str}")
        subprocess.run(full_cmd_list)

    def _initialize_bins(self):
        """Create bins ranging from 0 B to 512 PiB"""

        tmp = np.logspace(
            start=0, stop=59, num=60, base=2, dtype=np.float64
        )  # 2^0 2^1 ... 2^59
        self.bin_full = np.insert(tmp, 0, 0.0)  # 0 2^0 2^1 ... 2^59
        self.bin_full_in_MiB = self.bin_full / 1024.0 / 1024.0

    def _sql_query(self, filter_string: str) -> pd.DataFrame:
        """Perform SQL query.
        The SQLite schema in nsys is not forward compatible, and may change completely
        in a new release. Refer to
        https://docs.nvidia.com/nsight-systems/UserGuide/index.html?highlight=schema#sqlite-schema-reference

        :param filter_string: NVTX annotation string serving as a filter for the query.
        :type filter_string: str
        :return: Pandas dataframe containing the SQL query result.
        :rtype: pd.DataFrame
        """

        sql_expr = (
            "WITH io_string AS ( "
            + "    SELECT * "
            + "    FROM "
            + "        StringIds "
            + "    WHERE "
            + "        value LIKE '%%{}%%' ".format(filter_string)
            + "), "
            + "io_marker AS ( "
            + "    SELECT "
            + "        start AS startTimeInNs, "
            + "        int64Value AS ioSize, "
            + "        value AS nvtxAnnotation "
            + "    FROM NVTX_EVENTS "
            + "    CROSS JOIN io_string "
            + "    WHERE textId = io_string.id "
            + "    ORDER BY start "
            + ") "
            + "SELECT * "
            + "FROM io_marker;"
        )

        df = pd.read_sql(sql_expr, self.db_connection)
        return df

    def _generate_hist(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """_summary_

        :param df: _description_
        :type df: pd.DataFrame
        :return: _description_
        :rtype: tuple[np.ndarray, np.ndarray]
        """
        my_series = df["ioSize"]

        # Determine the appropriate bins for the histogram
        idx_upperbound = -1
        max_v = np.amax(my_series)
        for idx in range(len(self.bin_full_in_MiB)):
            if self.bin_full_in_MiB[idx] >= max_v:
                idx_upperbound = idx
                break

        tight_bin_edges = self.bin_full_in_MiB[0 : (idx_upperbound + 1)]
        if max_v > self.bin_full_in_MiB[-1]:
            tight_bin_edges.append(max_v)
        return np.histogram(my_series, tight_bin_edges)

    def _get_compact_filesize(self, file_size_inB: np.float64) -> str:
        """_summary_

        :param file_size_inB: _description_
        :type file_size_inB: np.float64
        :raises Exception: _description_
        :return: _description_
        :rtype: str
        """
        KiB = 1024.0
        MiB = 1024.0 * KiB
        GiB = 1024.0 * MiB
        TiB = 1024.0 * GiB
        PiB = 1024.0 * TiB
        EiB = 1024.0 * PiB

        if file_size_inB >= 0 and file_size_inB < KiB:
            return f"{int(file_size_inB)} B"
        elif file_size_inB >= KiB and file_size_inB < MiB:
            return f"{int(file_size_inB / KiB)} KiB"
        elif file_size_inB >= MiB and file_size_inB < GiB:
            return f"{int(file_size_inB / MiB)} MiB"
        elif file_size_inB >= GiB and file_size_inB < TiB:
            return f"{int(file_size_inB / GiB)} GiB"
        elif file_size_inB >= TiB and file_size_inB < PiB:
            return f"{int(file_size_inB / TiB)} TiB"
        elif file_size_inB >= PiB and file_size_inB < EiB:
            return f"{int(file_size_inB / PiB)} PiB"
        else:
            raise Exception("Invalid value for file_size.")

    def _print(self, title, hist, bin_edges):
        """_summary_

        :param title: _description_
        :type title: _type_
        :param hist: _description_
        :type hist: _type_
        :param bin_edges: _description_
        :type bin_edges: _type_
        """
        print(f"\n{title}")
        print("    Bins                 ...... Count")
        for idx in range(len(hist)):
            symbol = ")"
            if idx == len(hist) - 1:
                symbol = "]"

            print(
                "    [{:>8}, {:>8}{} ...... {}".format(
                    self._get_compact_filesize(bin_edges[idx]),
                    self._get_compact_filesize(bin_edges[idx + 1]),
                    symbol,
                    hist[idx],
                )
            )

    def _process(self, filter_string: str):
        """_summary_

        :param filter_string: _description_
        :type filter_string: str
        """
        df = self._sql_query(filter_string)
        if df.empty:
            print(f"\n{filter_string}")
            print("    Data is not detected.")
            return

        hist, bin_edges = self._generate_hist(df)
        self._print(filter_string, hist, bin_edges)

    def run(self):
        """_summary_"""
        self._initialize_bins()

        self._export_report_to_sqlite()
        self.db_connection = sqlite3.connect(self.sql_path)

        filter_string_list = [
            # Size of the read to be performed by KvikIO in parallel.
            # Source is the file, and destination is the host or device memory.
            "FileHandle::pread()",
            # Size of the write to be performed by KvikIO in parallel.
            # Source is the host or device memory, and destination is the file.
            "FileHandle::pwrite()",
            # Size of the read to be iteratively processed by POSIX pread() and CUDA
            # H2D memory copy.
            # Source is the file, and destination is the device memory.
            # This can be the individual task size as a result of KvikIO's
            # parallelization.
            # This can also be a simple sequential read size.
            "posix_device_read()",
            # Size of the write to be iteratively processed by CUDA D2H memory copy
            # and POSIX pwrite().
            # Source is the device memory, and destination is the file.
            # This can be the individual task size as a result of KvikIO's
            # parallelization.
            # This can also be a simple sequential write size.
            "posix_device_write()",
            # Size of the read to be iteratively processed by POSIX pread().
            # Source is the file, and destination is the host memory.
            # This can be the individual task size as a result of KvikIO's
            # parallelization.
            "posix_host_read()",
            # Size of the write to be iteratively processed by POSIX pwrite().
            # Source is the host memory, and destination is the file.
            # This can be the individual task size as a result of KvikIO's
            # parallelization.
            "posix_host_write()",
            # Size of the read passed to cuFile API.
            # Source is the file, and destination is the device memory.
            "cufileRead()",
            # Size of the write passed to cuFile API.
            # Source is the device memory, and destination is the file.
            "cufileWrite()",
            "RemoteHandle::read()",
            "RemoteHandle::pread()",
            "RemoteHandle - callback_host_memory()",
            "RemoteHandle - callback_device_memory()",
        ]

        for filter_string in filter_string_list:
            self._process(filter_string)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="kvikio_stat",
        description="Generate I/O size histogram from Nsight System report.",
    )
    parser.add_argument(
        "--nsys-report-path",
        required=True,
        help="The path of the Nsight System report.",
        type=str,
    )
    parser.add_argument(
        "--sql-path",
        help="kvikio_stat will invoke Nsight System to generated a SQL database from "
        + "the provided Nsight System report. --sql-path specifies the path of this "
        + "SQL database. If unspecified, the current working directory will be used "
        + "to store it, and the file name will be derived from the Nsight System "
        + "report.",
        type=str,
    )
    parser.add_argument(
        "--nsys-binary",
        help='The path of the Nsight System CLI program. If unspecified, "nsys" will '
        "be used.",
        type=str,
    )
    args = parser.parse_args()

    az = Analyzer(args)
    az.run()