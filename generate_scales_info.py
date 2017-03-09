#! /usr/bin/env python3
#
# Copyright (c) 2016, 2017, Forschungszentrum Juelich GmbH
# Author: Yann Leprince <y.leprince@fz-juelich.de>
#
# This software is made available under the MIT licence, see LICENCE.txt.


import copy
import json
import math
import os
import os.path

import numpy as np


def create_info_json_scales(info, target_chunk_size=64):
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
      "type": "image",
      "data_type": "uint8",
      "num_channels": 1,
      "scales": [
        {
          "chunk_sizes": [[64, 64, 64]],
          "encoding": "jpeg",
          "key": "full",
          "resolution": [20000, 20000, 20000],
          "size": [6572, 7404, 5711],
          "voxel_offset": [0, 0, 0]
        }
      ]
    }

    """
    assert len(info["scales"]) == 1
    full_scale_info = info["scales"][0]

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
        scale_info["key"] = str(round(min(res / 1000 for res in
                                          scale_info["resolution"]))) + "um"

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
    info["scales"] = [downscale_info(scale_level)
                      for scale_level in range(max_downscale_level)]
    return info


def generate_scales_info(input_fullres_info_filename):
    """Generate a list of scales from an input JSON file"""
    with open(input_fullres_info_filename) as f:
        info = json.load(f)
    create_info_json_scales(info)
    with open("info", "w") as f:
        json.dump(info, f)


def parse_command_line(argv):
    """Parse the script's command line."""
    import argparse
    parser = argparse.ArgumentParser(
        description="""\
Create a list of scales in Neuroglancer "info" JSON file format

Output is written to a file named "info" in the current directory.
""")
    parser.add_argument("input_fullres_info",
                       help="JSON file containing the full-resolution info")
    args = parser.parse_args(argv[1:])
    return args


def main(argv):
    """The script's entry point."""
    args = parse_command_line(argv)
    return linearize(args.fullres_info) or 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
