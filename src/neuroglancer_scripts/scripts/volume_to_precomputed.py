#! /usr/bin/env python3
#
# Copyright (c) 2016, 2017, Forschungszentrum Juelich GmbH
# Author: Yann Leprince <y.leprince@fz-juelich.de>
#
# This software is made available under the MIT licence, see LICENCE.txt.

import sys

import neuroglancer_scripts.accessor
import neuroglancer_scripts.chunk_encoding
import neuroglancer_scripts.volume_reader


def parse_command_line(argv):
    """Parse the script's command line."""
    import argparse
    parser = argparse.ArgumentParser(
        description="""\
Convert a volume from Nifti to Neuroglancer pre-computed format

Chunks are saved with the same data orientation as the input volume.

The image values will be scaled (additionally to any slope/intercept scaling
defined in the file header) if --input-max is specified. If --input-min is
omitted, it is assumed to be zero.
""")
    parser.add_argument("volume_filename",
                        help="source Nifti file containing the data")
    parser.add_argument("dest_url", help="directory/URL where the converted "
                        "dataset will be written")

    parser.add_argument("--generate-info", action="store_true",
                        help="generate an 'info_fullres.json' file containing "
                        "the metadata read for this volume, then exit")

    group = parser.add_argument_group("Option for reading the input file")
    group.add_argument("--ignore-scaling", action="store_true",
                       help="read the values as stored on disk, without "
                       "applying the data scaling (slope/intercept) from the "
                       "volume header")
    group.add_argument("--load-full-volume", action="store_true", default=True,
                       help=argparse.SUPPRESS)
    group.add_argument("--mmap", dest="load_full_volume", action="store_false",
                       help="use memory-mapping to avoid loading the full "
                       "volume in memory. This is useful if the input volume "
                       "is too large to fit memory, but it will slow down "
                       "the conversion significantly.")

    # TODO split into a module
    group = parser.add_argument_group(
        "Options for data type conversion and scaling")
    group.add_argument("--input-min", type=float, default=None,
                       help="input value that will be mapped to the minimum "
                       "output value")
    group.add_argument("--input-max", type=float, default=None,
                       help="input value that will be mapped to the maximum "
                       "output value")

    neuroglancer_scripts.accessor.add_argparse_options(parser)
    neuroglancer_scripts.chunk_encoding.add_argparse_options(parser,
                                                             allow_lossy=False)

    args = parser.parse_args(argv[1:])

    if args.input_max is None and args.input_min is not None:
        parser.error("--input-min cannot be specified if --input-max is "
                     "omitted")

    return args


def main(argv=sys.argv):
    """The script's entry point."""
    import neuroglancer_scripts.utils
    neuroglancer_scripts.utils.init_logging_for_cmdline()
    args = parse_command_line(argv)
    if args.generate_info:
        return neuroglancer_scripts.volume_reader.volume_file_to_info(
            args.volume_filename,
            args.dest_url,
            ignore_scaling=args.ignore_scaling,
            input_min=args.input_min,
            input_max=args.input_max,
            options=vars(args)
        ) or 0
    else:
        return neuroglancer_scripts.volume_reader.volume_file_to_precomputed(
            args.volume_filename,
            args.dest_url,
            ignore_scaling=args.ignore_scaling,
            input_min=args.input_min,
            input_max=args.input_max,
            load_full_volume=args.load_full_volume,
            options=vars(args)
        ) or 0


if __name__ == "__main__":
    sys.exit(main())
