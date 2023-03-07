#! /usr/bin/env python3
#
# Copyright (c) 2016–2018, Forschungszentrum Jülich GmbH
# Author: Yann Leprince <y.leprince@fz-juelich.de>
#
# This software is made available under the MIT licence, see LICENCE.txt.

import json
import logging
import sys

import nibabel

import neuroglancer_scripts.accessor
import neuroglancer_scripts.chunk_encoding
import neuroglancer_scripts.downscaling
import neuroglancer_scripts.dyadic_pyramid
import neuroglancer_scripts.scripts.generate_scales_info
from neuroglancer_scripts import precomputed_io
from neuroglancer_scripts import volume_reader


logger = logging.getLogger(__name__)


def volume_to_precomputed_pyramid(volume_filename,
                                  dest_url,
                                  downscaling_method="average",
                                  ignore_scaling=False,
                                  input_min=None,
                                  input_max=None,
                                  load_full_volume=True,
                                  dataset_type=None,
                                  encoding=None,
                                  options={}):
    img = nibabel.load(volume_filename)
    formatted_info, _, _, _ = volume_reader.nibabel_image_to_info(
        img,
        ignore_scaling=ignore_scaling,
        input_min=input_min,
        input_max=input_max,
        options=options
    )
    info = json.loads(formatted_info)
    accessor = neuroglancer_scripts.accessor.get_accessor_for_url(
        dest_url, options
    )
    neuroglancer_scripts.scripts.generate_scales_info.set_info_params(
        info,
        dataset_type=dataset_type,
        encoding=encoding
    )
    neuroglancer_scripts.dyadic_pyramid.fill_scales_for_dyadic_pyramid(
        info
    )
    try:
        precomputed_writer = precomputed_io.get_IO_for_new_dataset(
            info, accessor
        )
    except neuroglancer_scripts.accessor.DataAccessError as exc:
        logger.error("Cannot write info: {0}".format(exc))
        return 1
    volume_reader.nibabel_image_to_precomputed(
        img, precomputed_writer,
        ignore_scaling, input_min, input_max,
        load_full_volume, options
    )
    downscaler = neuroglancer_scripts.downscaling.get_downscaler(
        downscaling_method, info, options
    )
    neuroglancer_scripts.dyadic_pyramid.compute_dyadic_scales(
        precomputed_writer, downscaler
    )


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
    group.add_argument("--type", default=None,
                        choices=("image", "segmentation"),
                        help="Type of dataset (image or segmentation). By"
                        " default this is inherited from the fullres_info"
                        " file, with a fallback to image.")
    group.add_argument("--encoding", default=None,
                        choices=("raw", "jpeg", "compressed_segmentation"),
                        help="data encoding (raw, jpeg, or"
                        " compressed_segmentation). By default this is"
                        " inherited from the fullres_info file, with a"
                        " fallback to raw.")
    
    neuroglancer_scripts.accessor.add_argparse_options(parser)
    neuroglancer_scripts.downscaling.add_argparse_options(parser)
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
    return volume_to_precomputed_pyramid(
        args.volume_filename,
        args.dest_url,
        downscaling_method=args.downscaling_method,
        ignore_scaling=args.ignore_scaling,
        input_min=args.input_min,
        input_max=args.input_max,
        load_full_volume=args.load_full_volume,
        dataset_type=args.type,
        encoding=args.encoding,
        options=vars(args)
    ) or 0


if __name__ == "__main__":
    sys.exit(main())
