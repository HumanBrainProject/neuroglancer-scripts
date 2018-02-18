#! /usr/bin/env python3
#
# Copyright (c) 2016, 2017, Forschungszentrum Juelich GmbH
# Author: Yann Leprince <y.leprince@fz-juelich.de>
#
# This software is made available under the MIT licence, see LICENCE.txt.


import collections
import copy
import json
import math
import sys

import neuroglancer_scripts.accessor
from neuroglancer_scripts import precomputed_io


KEY_UNITS = collections.OrderedDict([
    ("km", 1e-12),
    ("m", 1e-9),
    ("mm", 1e-6),
    ("um", 1e-3),
    ("nm", 1.),
    ("pm", 1e3),
])


def format_key(resolution_nm, unit):
    return format(resolution_nm * KEY_UNITS[unit], ".0f") + unit


def choose_unit_for_key(resolution_nm):
    """Find the coarsest unit that ensures distinct keys"""
    for unit, factor in KEY_UNITS.items():
        if (round(resolution_nm * factor) != 0
            and (format_key(resolution_nm, unit)
                 != format_key(resolution_nm * 2, unit))):
            return unit
    raise RuntimeError("cannot find a suitable unit for {} nm"
                       .format(resolution_nm))


def create_info_json_scales(info, target_chunk_size=64,
                            dataset_type=None,
                            encoding=None, max_scales=None):
    """Create a list of scales in Neuroglancer "info" JSON file format

    This function takes a dictionary as input, in the format documented at
    https://github.com/google/neuroglancer/blob/master/src/neuroglancer/datasource/precomputed/README.md.
    The "scales" parameter must contain only one element, representing the full
    resolution (see example input below). The chunk_sizes element is ignored,
    and overwritten with a value derived from the target_chunk_size parameter.

    This function edits the input dictionary in-place, and also returns it as
    an output. It populates the "scales" element with lower-resolution volumes,
    which are downscaled from the full resolution by an integer power of two.
    Anisotropic volumes are handled in such a way, that the downscaled versions
    approach an isotropic voxel size. The chunk size is adapted from the
    target_chunk_size so that the spatial extent of a chunk, and the voxel
    count, are approximately constant. target_chunk_size must be an integer
    power of two.

    Example input (for the BigBrain):
    {
      "data_type": "uint8",
      "num_channels": 1,
      "scales": [
        {
          "resolution": [20000, 20000, 20000],
          "size": [6572, 7404, 5711],
          "voxel_offset": [0, 0, 0]
        }
      ]
    }

    """
    if len(info["scales"]) != 1:
        print("WARNING: the source info JSON contains multiple scales, only "
              "the first one will be used.")
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
            print("WARNING: using compressed_segmentation encoding but "
                  "'type' is not 'segmentation'")
        if "compressed_segmentation_block_size" not in full_scale_info:
            full_scale_info["compressed_segmentation_block_size"] = [8, 8, 8]
        # compressed_segmentation only supports uint32 or uint64
        if info["data_type"] in ("uint8", "uint16"):
            info["data_type"] = "uint32"

    target_chunk_exponent = int(math.log2(target_chunk_size))
    assert target_chunk_size == 2 ** target_chunk_exponent

    # The concept of “scale level”, or just “level” is used here: it is an
    # integer that represents the number of downscaling steps since the
    # full-resolution volume. Each downscaling step reduces the dimension by a
    # factor of two in all axes (except in the case of an anisotropic volume,
    # where the downscaling of some axes is delayed until the voxel size is
    # closer to isotropic, this is the role of the axis_level_delays list).

    full_resolution = full_scale_info["resolution"]
    best_axis_resolution = min(full_resolution)
    axis_level_delays = [
        int(round(math.log2(axis_res / best_axis_resolution)))
        for axis_res in full_resolution]
    key_unit = choose_unit_for_key(best_axis_resolution)

    def downscale_info(scale_level):
        factors = [2 ** max(0, scale_level - delay)
                   for delay in axis_level_delays]
        scale_info = copy.deepcopy(full_scale_info)
        scale_info["resolution"] = [
            res * axis_factor for res, axis_factor in
            zip(full_scale_info["resolution"], factors)]
        scale_info["size"] = [
            (sz - 1) // axis_factor + 1 for sz, axis_factor in
            zip(full_scale_info["size"], factors)]
        # Key is the resolution in micrometres
        scale_info["key"] = format_key(min(scale_info["resolution"]), key_unit)

        max_delay = max(axis_level_delays)
        anisotropy_factors = [max(0, max_delay - delay - scale_level)
                              for delay in axis_level_delays]
        sum_anisotropy_factors = sum(anisotropy_factors)
        base_chunk_exponent = (
            target_chunk_exponent - (sum_anisotropy_factors + 1) // 3)
        scale_info["chunk_sizes"] = [
            [2 ** (base_chunk_exponent + anisotropy_factor)
             for anisotropy_factor in anisotropy_factors]]

        assert (abs(sum(int(round(math.log2(size)))
                        for size in scale_info["chunk_sizes"][0])
                    - 3 * target_chunk_exponent) <= 1)

        return scale_info

    # Stop when the downscaled volume fits in two chunks (is target_chunk_size
    # adequate, or should we use the actual chunk sizes?)
    max_downscale_level = (
        max(math.ceil(math.log2(a / target_chunk_size)) - b
            for a, b in zip(full_scale_info["size"],
                            axis_level_delays)))
    if max_scales:
        max_downscale_level = min(max_downscale_level, max_scales)
    info["scales"] = [downscale_info(scale_level)
                      for scale_level in range(max_downscale_level)]
    return info


def generate_scales_info(input_fullres_info_filename,
                         dest_url,
                         target_chunk_size=64,
                         dataset_type=None,
                         encoding=None,
                         max_scales=None):
    """Generate a list of scales from an input JSON file"""
    with open(input_fullres_info_filename) as f:
        info = json.load(f)
    create_info_json_scales(info, target_chunk_size=target_chunk_size,
                            dataset_type=dataset_type, encoding=encoding,
                            max_scales=max_scales)
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
    args = parse_command_line(argv)
    return generate_scales_info(args.fullres_info,
                                args.dest_url,
                                target_chunk_size=args.target_chunk_size,
                                dataset_type=args.type,
                                encoding=args.encoding,
                                max_scales=args.max_scales) or 0


if __name__ == "__main__":
    sys.exit(main())
