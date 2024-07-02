#! /usr/bin/env python3
#
# Copyright (c) 2016, 2017, 2023 Forschungszentrum Juelich GmbH
# Author: Yann Leprince <y.leprince@fz-juelich.de>
# Author: Xiao Gui <xgui3783@gmail.com>
#
# This software is made available under the MIT licence, see LICENCE.txt.

import sys

import numpy as np

import neuroglancer_scripts.accessor
from neuroglancer_scripts import precomputed_io
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

        shard_info = "Unsharded"
        shard_spec = scale.get("sharding")
        sharding_num_directories = None
        if shard_spec:
            shard_bits = shard_spec.get("shard_bits")
            shard_info = f"Sharded: {shard_bits}bits"
            sharding_num_directories = 2 ** shard_bits + 1

        for chunk_size in scale["chunk_sizes"]:
            size_in_chunks = [(s - 1) // cs + 1 for s,
                              cs in zip(size, chunk_size)]
            num_chunks = np.prod(size_in_chunks)
            num_directories = (
                sharding_num_directories
                if sharding_num_directories is not None
                else size_in_chunks[0] * (1 + size_in_chunks[1]))
            size_bytes = np.prod(size) * dtype.itemsize * num_channels
            print(f"Scale {scale_name}, {shard_info}, chunk size {chunk_size}:"
                  f" {num_chunks:,d} chunks, {num_directories:,d} directories,"
                  f" raw uncompressed size {readable_count(size_bytes)}B")
            total_size += size_bytes
            total_chunks += num_chunks
            total_directories += num_directories
    print("---")
    print(f"Total: {total_chunks:,d} chunks, {total_directories:,d} "
          f"directories, raw uncompressed size {readable_count(total_size)}B")


def show_scale_file_info(url, options={}):
    """Show information about a list of scales."""
    accessor = neuroglancer_scripts.accessor.get_accessor_for_url(url, options)
    io = precomputed_io.get_IO_for_existing_dataset(accessor)
    info = io.info
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

    neuroglancer_scripts.accessor.add_argparse_options(
        parser, write_chunks=False, write_files=False
    )
    args = parser.parse_args(argv[1:])
    return args


def main(argv=sys.argv):
    """The script's entry point."""
    import neuroglancer_scripts.utils
    neuroglancer_scripts.utils.init_logging_for_cmdline()
    args = parse_command_line(argv)
    return show_scale_file_info(args.url, options=vars(args)) or 0


if __name__ == "__main__":
    sys.exit(main())
