#! /usr/bin/env python3
#
# Copyright (c) 2016, 2017, 2018 Forschungszentrum Juelich GmbH
# Author: Yann Leprince <y.leprince@fz-juelich.de>
#
# This software is made available under the MIT licence, see LICENCE.txt.

import sys

import numpy as np
from tqdm import tqdm

import neuroglancer_scripts.accessor
import neuroglancer_scripts.chunk_encoding
from neuroglancer_scripts import data_types
from neuroglancer_scripts import precomputed_io


def convert_chunks_for_scale(chunk_reader,
                             dest_info, chunk_writer, scale_index,
                             chunk_transformer):
    """Convert chunks for a given scale"""
    scale_info = dest_info["scales"][scale_index]
    key = scale_info["key"]
    size = scale_info["size"]
    dest_dtype = np.dtype(dest_info["data_type"]).newbyteorder("<")

    for chunk_size in scale_info["chunk_sizes"]:
        chunk_range = ((size[0] - 1) // chunk_size[0] + 1,
                       (size[1] - 1) // chunk_size[1] + 1,
                       (size[2] - 1) // chunk_size[2] + 1)
        for x_idx, y_idx, z_idx in tqdm(
                np.ndindex(chunk_range), total=np.prod(chunk_range),
                unit="chunk",
                desc="converting scale {}".format(key)):
            xmin = chunk_size[0] * x_idx
            xmax = min(chunk_size[0] * (x_idx + 1), size[0])
            ymin = chunk_size[1] * y_idx
            ymax = min(chunk_size[1] * (y_idx + 1), size[1])
            zmin = chunk_size[2] * z_idx
            zmax = min(chunk_size[2] * (z_idx + 1), size[2])
            chunk_coords = (xmin, xmax, ymin, ymax, zmin, zmax)

            chunk = chunk_reader.read_chunk(key, chunk_coords)
            chunk_writer.write_chunk(
                chunk.astype(dest_dtype, casting="equiv"),
                key, chunk_coords
            )


def convert_chunks(source_url, dest_url, copy_info=False,
                   options={}):
    """Convert precomputed chunks between different encodings"""
    source_accessor = neuroglancer_scripts.accessor.get_accessor_for_url(
        source_url
    )
    chunk_reader = precomputed_io.get_IO_for_existing_dataset(source_accessor)
    source_info = chunk_reader.info
    dest_accessor = neuroglancer_scripts.accessor.get_accessor_for_url(
        dest_url, options
    )
    if copy_info:
        chunk_writer = precomputed_io.get_IO_for_new_dataset(
            source_info, dest_accessor, encoder_options=options
        )
    else:
        chunk_writer = precomputed_io.get_IO_for_existing_dataset(
            dest_accessor, encoder_options=options
        )
    dest_info = chunk_writer.info

    chunk_transformer = data_types.get_chunk_dtype_transformer(
        source_info["data_type"], dest_info["data_type"]
    )
    for scale_index in reversed(range(len(dest_info["scales"]))):
        convert_chunks_for_scale(chunk_reader,
                                 dest_info, chunk_writer, scale_index,
                                 chunk_transformer)


def parse_command_line(argv):
    """Parse the script's command line."""
    import argparse
    parser = argparse.ArgumentParser(
        description="""\
Convert Neuroglancer precomputed chunks between different encodings (raw,
compressed_segmentation, or jpeg). The target encoding parameters is determined
by a pre-existing info file in the destination directory (except in --copy-info
mode). You can create such an info file with generate_scales_info.py.
""")
    parser.add_argument("source_url",
                        help="URL/directory where the input chunks are found")
    parser.add_argument("dest_url", default=".",
                        help="URL/directory where the output chunks will be "
                        "written.")

    parser.add_argument("--copy-info", action="store_true",
                        help="Copy the info file instead of using a "
                        "pre-existing. The data will be re-encoded with the "
                        "same encoding as the original")

    neuroglancer_scripts.accessor.add_argparse_options(parser)
    neuroglancer_scripts.chunk_encoding.add_argparse_options(parser,
                                                             allow_lossy=True)

    args = parser.parse_args(argv[1:])
    return args


def main(argv=sys.argv):
    """The script's entry point."""
    import neuroglancer_scripts.utils
    neuroglancer_scripts.utils.init_logging_for_cmdline()
    args = parse_command_line(argv)
    return convert_chunks(args.source_url, args.dest_url,
                          copy_info=args.copy_info,
                          options=vars(args)) or 0


if __name__ == "__main__":
    sys.exit(main())
