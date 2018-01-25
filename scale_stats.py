#! /usr/bin/env python3
#
# Copyright (c) 2016, 2017, Forschungszentrum Juelich GmbH
# Author: Yann Leprince <y.leprince@fz-juelich.de>
#
# This software is made available under the MIT licence, see LICENCE.txt.


import collections
import json
import math
import os
import os.path
import sys

import numpy as np

SI_PREFIXES = [
    (1, ""),
    (1024, "ki"),
    (1024 * 1024, "Mi"),
    (1024 * 1024 * 1024, "Gi"),
    (1024 * 1024 * 1024 * 1024, "Ti"),
    (1024 * 1024 * 1024 * 1024 * 1024, "Pi"),
    (1024 * 1024 * 1024 * 1024 * 1024 * 1024, "Ei"),
]


def readable(count):
    for factor, prefix in SI_PREFIXES:
        if count > 10 * factor:
            num_str = format(count / factor, ".0f")
        else:
            num_str = format(count / factor, ".1f")
        if len(num_str) <= 3:
            return num_str + " " + prefix
    # Fallback: use the last prefix
    factor, prefix = SI_PREFIXES[-1]
    return "{:,.0f} {}".format(count / factor, prefix)


def show_scales_info(info):
    total_size = 0
    total_chunks = 0
    total_directories = 0
    dtype = np.dtype(info["data_type"]).newbyteorder("<")
    num_channels = info["num_channels"]
    for scale in info["scales"]:
        scale_name = scale["key"]
        size = scale["size"]
        for chunk_size in scale["chunk_sizes"]:
            size_in_chunks = [(s - 1) // cs + 1 for s,
                              cs in zip(size, chunk_size)]
            num_chunks = np.prod(size_in_chunks)
            num_directories = size_in_chunks[0] * (1 + size_in_chunks[1])
            size_bytes = np.prod(size) * dtype.itemsize * num_channels
            print("Scale {}, chunk size {}:"
                  " {:,d} chunks, {:,d} directories, raw uncompressed size {}B"
                  .format(scale_name, chunk_size,
                          num_chunks, num_directories, readable(size_bytes)))
            total_size += size_bytes
            total_chunks += num_chunks
            total_directories += num_directories
    print("---")
    print("Total: {:,d} chunks, {:,d} directories, raw uncompressed size {}B"
          .format(total_chunks, total_directories, readable(total_size)))


def show_scale_file_info(input_info_filename):
    """Show information about a list of scales from an input JSON file"""
    with open(input_info_filename) as f:
        info = json.load(f)
    show_scales_info(info)


def parse_command_line(argv):
    """Parse the script's command line."""
    import argparse
    parser = argparse.ArgumentParser(
        description="""\
Show information about a list of scales in Neuroglancer "info" JSON file format
""")
    parser.add_argument("info_file", nargs="?", default="./info",
                        help="JSON file containing the information")
    args = parser.parse_args(argv[1:])
    return args


def main(argv):
    """The script's entry point."""
    args = parse_command_line(argv)
    return show_scale_file_info(args.info_file) or 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
