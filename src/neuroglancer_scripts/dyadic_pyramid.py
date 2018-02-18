# Copyright (c) 2016–2018, Forschungszentrum Jülich GmbH
# Author: Yann Leprince <y.leprince@fz-juelich.de>
#
# This software is made available under the MIT licence, see LICENCE.txt.

import copy
import logging
import math

from neuroglancer_scripts.utils import (LENGTH_UNITS, format_length)


logger = logging.getLogger(__name__)


def choose_unit_for_key(resolution_nm):
    """Find the coarsest unit that ensures distinct keys"""
    for unit, factor in LENGTH_UNITS.items():
        if (round(resolution_nm * factor) != 0
            and (format_length(resolution_nm, unit)
                 != format_length(resolution_nm * 2, unit))):
            return unit
    raise RuntimeError("cannot find a suitable unit for {} nm"
                       .format(resolution_nm))


def fill_scales_for_dyadic_pyramid(info, target_chunk_size=64,
                                   max_scales=None):
    """Set up a dyadic pyramid in an *info* file.

    :param dict info: input *info* with only one scale. It is **modified
                      in-place** to add the list of dyadic scales.
    :param int target_chunk_size: see above, must be an integer power of two
    :param int max_scales: maximum number of scales in the output *info*

    The "scales" parameter is assumed to contain only one element, representing
    the full resolution (see example input below). The chunk_sizes element is
    ignored, and overwritten with a value derived from the target_chunk_size
    parameter.

    This function edits the input dictionary in-place, and also returns it as
    an output. It populates the "scales" element with lower-resolution volumes,
    which are downscaled from the full resolution by an integer power of two.

    Isotropic volumes will have chunks of size :arg:`target_chunk_size` in all
    three dimensions. Each scale will downscale the volume by a factor of two
    in all three dimensions.

    For anisotropic volumes, only the dimensions with the smallest voxel size
    are downscaled until the downscaled voxels are as close to isotropic as
    possible. The voxels are then downscaled in all three directions, as for
    isotropic volumes. The chunk size will be adapted so that the chunks have a
    spatial extent which is close to isotropic, while keeping the total number
    of voxels in a chunk close to ``target_chunk_size**3 ``.

    The number of scales is so that until the final downscaled volume fits in
    at most two chunks along every dimension.
    """
    if len(info["scales"]) != 1:
        logger.warn("the source info JSON contains multiple scales, only "
                    "the first one will be used.")
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
        scale_info["key"] = format_length(min(scale_info["resolution"]),
                                          key_unit)

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
