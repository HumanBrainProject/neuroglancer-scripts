#! /usr/bin/env python3
#
# Copyright (c) 2016, 2017, Forschungszentrum Juelich GmbH
# Author: Yann Leprince <y.leprince@fz-juelich.de>
#
# This software is made available under the MIT licence, see LICENCE.txt.


import json
import logging
import sys

import neuroglancer_scripts.accessor
from neuroglancer_scripts import data_types
from neuroglancer_scripts import precomputed_io
import neuroglancer_scripts.dyadic_pyramid

logger = logging.getLogger(__name__)


def set_info_params(info, dataset_type=None, encoding=None):
    full_scale_info = info["scales"][0]
    if encoding:
        full_scale_info["encoding"] = encoding
    elif "encoding" not in full_scale_info:
        full_scale_info["encoding"] = "raw"

    if dataset_type:
        info["type"] = dataset_type
    elif "type" not in info:
        if full_scale_info["encoding"] == "compressed_segmentation":
            info["type"] = "segmentation"
        else:
            info["type"] = "image"

    if full_scale_info["encoding"] == "compressed_segmentation":
        if info["type"] != "segmentation":
            logger.warn("using compressed_segmentation encoding but "
                        "'type' is not 'segmentation'")
        if "compressed_segmentation_block_size" not in full_scale_info:
            full_scale_info["compressed_segmentation_block_size"] = [8, 8, 8]
        # compressed_segmentation only supports uint32 or uint64
        if info["data_type"] in ("uint8", "uint16"):
            info["data_type"] = "uint32"
        if info["data_type"] not in ("uint32", "uint64"):
            logger.warn("data type %s is not supported by the "
                        "compressed_segmentation encoding",
                        info["data_type"])

    if (info["type"] == "segmentation"
            and info["data_type"] not in data_types.NG_INTEGER_DATA_TYPES):
        logger.warn('the dataset is of type "segmentation" but has a '
                    'non-integer data_type (%s)', info["data_type"])


def generate_scales_info(input_fullres_info_filename,
                         dest_url,
                         target_chunk_size=64,
                         dataset_type=None,
                         encoding=None,
                         max_scales=None):
    """Generate a list of scales from an input JSON file"""
    with open(input_fullres_info_filename) as f:
        info = json.load(f)
    set_info_params(info, dataset_type=dataset_type, encoding=encoding)
    neuroglancer_scripts.dyadic_pyramid.fill_scales_for_dyadic_pyramid(
        info,
        target_chunk_size=target_chunk_size,
        max_scales=max_scales
    )
    accessor = neuroglancer_scripts.accessor.get_accessor_for_url(dest_url)
    # This writes the info file
    precomputed_io.get_IO_for_new_dataset(info, accessor)


def parse_command_line(argv):
    """Parse the script's command line."""
    import argparse
    parser = argparse.ArgumentParser(
        description="""\
Create a list of scales in Neuroglancer "info" JSON file format.

Output is written to a file named "info" at dest_url.
""")
    parser.add_argument("fullres_info",
                        help="JSON file containing the full-resolution info")
    parser.add_argument("dest_url", help="directory/URL where the converted "
                        "dataset will be written")
    parser.add_argument("--max-scales", type=int, default=None,
                        help="maximum number of scales to generate"
                        " (default: unlimited)")
    parser.add_argument("--target-chunk-size", type=int, default=64,
                        help="target chunk size (default 64). This size will"
                        " be used for cubic chunks, the size of anisotropic"
                        " chunks will be adapted to contain approximately the"
                        " same number of voxels.")
    parser.add_argument("--type", default=None,
                        choices=("image", "segmentation"),
                        help="Type of dataset (image or segmentation). By"
                        " default this is inherited from the fullres_info"
                        " file, with a fallback to image.")
    parser.add_argument("--encoding", default=None,
                        choices=("raw", "jpeg", "compressed_segmentation"),
                        help="data encoding (raw, jpeg, or"
                        " compressed_segmentation). By default this is"
                        " inherited from the fullres_info file, with a"
                        " fallback to raw.")
    args = parser.parse_args(argv[1:])
    return args


def main(argv=sys.argv):
    """The script's entry point."""
    import neuroglancer_scripts.utils
    neuroglancer_scripts.utils.init_logging_for_cmdline()
    args = parse_command_line(argv)
    return generate_scales_info(args.fullres_info,
                                args.dest_url,
                                target_chunk_size=args.target_chunk_size,
                                dataset_type=args.type,
                                encoding=args.encoding,
                                max_scales=args.max_scales) or 0


if __name__ == "__main__":
    sys.exit(main())
