#! /usr/bin/env python3
#
# Copyright (c) 2016, 2017, 2018 Forschungszentrum Juelich GmbH
# Author: Yann Leprince <y.leprince@fz-juelich.de>
#
# This software is made available under the MIT licence, see LICENCE.txt.


import sys

import neuroglancer_scripts.accessor
import neuroglancer_scripts.chunk_encoding
import neuroglancer_scripts.dyadic_pyramid
import neuroglancer_scripts.downscaling
from neuroglancer_scripts import precomputed_io


def compute_scales(work_dir=".", downscaling_method="average", options={}):
    """Generate lower scales following an input info file"""
    accessor = neuroglancer_scripts.accessor.get_accessor_for_url(
        work_dir, options
    )
    pyramid_io = precomputed_io.get_IO_for_existing_dataset(
        accessor, encoder_options=options
    )
    downscaler = neuroglancer_scripts.downscaling.get_downscaler(
        downscaling_method, options)
    neuroglancer_scripts.dyadic_pyramid.compute_dyadic_scales(
        pyramid_io, downscaler
    )


def parse_command_line(argv):
    """Parse the script's command line."""
    import argparse
    parser = argparse.ArgumentParser(
        description="""\
Create lower scales in Neuroglancer precomputed format

The list of scales is read from a file named "info" in the working directory.
All the lower resolutions are computed from the the highest resolution (first
scale in the info file). Only downscaling by a factor of 2 is supported (any
pyramid scheme created by generate_scales_info.py is appropriate).
""")
    parser.add_argument("work_dir", help="working directory or URL")

    neuroglancer_scripts.accessor.add_argparse_options(parser)
    neuroglancer_scripts.downscaling.add_argparse_options(parser)
    neuroglancer_scripts.chunk_encoding.add_argparse_options(parser,
                                                             allow_lossy=False)

    args = parser.parse_args(argv[1:])
    return args


def main(argv=sys.argv):
    """The script's entry point."""
    args = parse_command_line(argv)
    return compute_scales(args.work_dir,
                          args.downscaling_method,
                          options=vars(args)) or 0


if __name__ == "__main__":
    sys.exit(main())
