#! /usr/bin/env python3
#
# Copyright (c) 2016, 2017, Forschungszentrum Juelich GmbH
# Author: Yann Leprince <y.leprince@fz-juelich.de>
#
# This software is made available under the MIT licence, see LICENCE.txt.

import os
import os.path
import sys

import numpy as np

import neuroglancer_scripts.accessor
from neuroglancer_scripts.utils import readable_count


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
                          num_chunks, num_directories,
                          readable_count(size_bytes)))
            total_size += size_bytes
            total_chunks += num_chunks
            total_directories += num_directories
    print("---")
    print("Total: {:,d} chunks, {:,d} directories, raw uncompressed size {}B"
          .format(total_chunks, total_directories,
                  readable_count(total_size)))


def show_scale_file_info(url, options={}):
    """Show information about a list of scales."""
    accessor = neuroglancer_scripts.accessor.get_accessor_for_url(url, options)
    info = accessor.fetch_info()
    show_scales_info(info)


def parse_command_line(argv):
    """Parse the script's command line."""
    import argparse
    parser = argparse.ArgumentParser(
        description="""\
Show information about a list of scales in Neuroglancer "info" JSON file format
""")
    parser.add_argument("url", default=".",
                        help='directory/URL containing the "info" file')

    neuroglancer_scripts.accessor.add_argparse_options(parser, write=False)
    args = parser.parse_args(argv[1:])
    return args


def main(argv=sys.argv):
    """The script's entry point."""
    args = parse_command_line(argv)
    return show_scale_file_info(args.url, options=vars(args)) or 0


if __name__ == "__main__":
    sys.exit(main())
