# Copyright (c) 2016–2018, Forschungszentrum Jülich GmbH
# Author: Yann Leprince <y.leprince@fz-juelich.de>
#
# This software is made available under the MIT licence, see LICENCE.txt.

import copy
import logging
import math

import numpy as np
from tqdm import tqdm

from neuroglancer_scripts.utils import LENGTH_UNITS, ceil_div, format_length

__all__ = [
    "choose_unit_for_key",
    "fill_scales_for_dyadic_pyramid",
    "compute_dyadic_scales",
    "compute_dyadic_downscaling",
]


logger = logging.getLogger(__name__)


def choose_unit_for_key(resolution_nm):
    """Find the coarsest unit that ensures distinct keys"""
    for unit, factor in LENGTH_UNITS.items():
        if (round(resolution_nm * factor) != 0
            and (format_length(resolution_nm, unit)
                 != format_length(resolution_nm * 2, unit))):
            return unit
    raise NotImplementedError("cannot find a suitable unit for {} nm"
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
        logger.warning("the source info JSON contains multiple scales, only "
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
            ceil_div(sz, axis_factor) for sz, axis_factor in
            zip(full_scale_info["size"], factors)]
        # Key is the resolution in micrometres
        scale_info["key"] = format_length(min(scale_info["resolution"]),
                                          key_unit)

        max_delay = max(axis_level_delays)
        anisotropy_factors = [max(0, max_delay - delay - scale_level)
                              for delay in axis_level_delays]
        sum_anisotropy_factors = sum(anisotropy_factors)

        # Ensure that the smallest chunk size is 1 for extremely anisotropic
        # datasets (i.e. reduce the anisotropy of chunk_size)
        excess_anisotropy = sum_anisotropy_factors - 3 * target_chunk_exponent
        if excess_anisotropy > 0:
            anisotropy_reduction = ceil_div(excess_anisotropy,
                                            sum(1 for f in anisotropy_factors
                                                if f != 0))
            anisotropy_factors = [max(f - anisotropy_reduction, 0)
                                  for f in anisotropy_factors]
            sum_anisotropy_factors = sum(anisotropy_factors)
            assert sum_anisotropy_factors <= 3 * target_chunk_exponent

        base_chunk_exponent = (
            target_chunk_exponent - (sum_anisotropy_factors + 1) // 3)
        assert base_chunk_exponent >= 0
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
    max_downscale_level = max(max_downscale_level, 1)
    info["scales"] = [downscale_info(scale_level)
                      for scale_level in range(max_downscale_level)]
    return info


def compute_dyadic_scales(precomputed_io, downscaler):
    from neuroglancer_scripts import sharded_file_accessor
    for i in range(len(precomputed_io.info["scales"]) - 1):
        compute_dyadic_downscaling(
            precomputed_io.info, i, downscaler, precomputed_io, precomputed_io
        )
        if isinstance(precomputed_io.accessor,
                      sharded_file_accessor.ShardedFileAccessor):
            precomputed_io.accessor.close()


def compute_dyadic_downscaling(info, source_scale_index, downscaler,
                               chunk_reader, chunk_writer):
    # Key is the resolution in micrometres
    old_scale_info = info["scales"][source_scale_index]
    new_scale_info = info["scales"][source_scale_index + 1]
    old_chunk_size = old_scale_info["chunk_sizes"][0]
    new_chunk_size = new_scale_info["chunk_sizes"][0]
    old_key = old_scale_info["key"]
    new_key = new_scale_info["key"]
    old_size = old_scale_info["size"]
    new_size = new_scale_info["size"]
    dtype = np.dtype(info["data_type"]).newbyteorder("<")
    num_channels = info["num_channels"]
    downscaling_factors = [1 if os == ns else 2
                           for os, ns in zip(old_size, new_size)]
    if new_size != [ceil_div(os, ds)
                    for os, ds in zip(old_size, downscaling_factors)]:
        raise ValueError("Unsupported downscaling factor between scales "
                         "{} and {} (only 1 and 2 are supported)"
                         .format(old_key, new_key))

    downscaler.check_factors(downscaling_factors)

    if chunk_reader.scale_is_lossy(old_key):
        logger.warning(
            "Using data stored in a lossy format (scale %s) as an input "
            "for downscaling (to scale %s)" % (old_key, new_key)
        )

    half_chunk = [osz // f
                  for osz, f in zip(old_chunk_size, downscaling_factors)]
    chunk_fetch_factor = [nsz // hc
                          for nsz, hc in zip(new_chunk_size, half_chunk)]

    def load_and_downscale_old_chunk(z_idx, y_idx, x_idx):
        xmin = old_chunk_size[0] * x_idx
        xmax = min(old_chunk_size[0] * (x_idx + 1), old_size[0])
        ymin = old_chunk_size[1] * y_idx
        ymax = min(old_chunk_size[1] * (y_idx + 1), old_size[1])
        zmin = old_chunk_size[2] * z_idx
        zmax = min(old_chunk_size[2] * (z_idx + 1), old_size[2])
        old_chunk_coords = (xmin, xmax, ymin, ymax, zmin, zmax)

        chunk = chunk_reader.read_chunk(old_key, old_chunk_coords)

        return downscaler.downscale(chunk, downscaling_factors)

    chunk_range = (ceil_div(new_size[0], new_chunk_size[0]),
                   ceil_div(new_size[1], new_chunk_size[1]),
                   ceil_div(new_size[2], new_chunk_size[2]))
    # TODO how to do progress report correctly with logging?
    for x_idx, y_idx, z_idx in tqdm(
            np.ndindex(chunk_range), total=np.prod(chunk_range),
            desc="computing scale {}".format(new_key),
            unit="chunks", leave=True):
        xmin = new_chunk_size[0] * x_idx
        xmax = min(new_chunk_size[0] * (x_idx + 1), new_size[0])
        ymin = new_chunk_size[1] * y_idx
        ymax = min(new_chunk_size[1] * (y_idx + 1), new_size[1])
        zmin = new_chunk_size[2] * z_idx
        zmax = min(new_chunk_size[2] * (z_idx + 1), new_size[2])
        new_chunk_coords = (xmin, xmax, ymin, ymax, zmin, zmax)
        new_chunk = np.empty(
            [num_channels, zmax - zmin, ymax - ymin, xmax - xmin],
            dtype=dtype
        )
        new_chunk[:, :half_chunk[2], :half_chunk[1],
                  :half_chunk[0]] = (
                      load_and_downscale_old_chunk(
                          z_idx * chunk_fetch_factor[2],
                          y_idx * chunk_fetch_factor[1],
                          x_idx * chunk_fetch_factor[0]))
        if new_chunk.shape[1] > half_chunk[2]:
            new_chunk[:, half_chunk[2]:, :half_chunk[1],
                      :half_chunk[0]] = (
                          load_and_downscale_old_chunk(
                              z_idx * chunk_fetch_factor[2] + 1,
                              y_idx * chunk_fetch_factor[1],
                              x_idx * chunk_fetch_factor[0]))
        if new_chunk.shape[2] > half_chunk[1]:
            new_chunk[:, :half_chunk[2], half_chunk[1]:,
                      :half_chunk[0]] = (
                          load_and_downscale_old_chunk(
                              z_idx * chunk_fetch_factor[2],
                              y_idx * chunk_fetch_factor[1] + 1,
                              x_idx * chunk_fetch_factor[0]))
        if (new_chunk.shape[1] > half_chunk[2]
                and new_chunk.shape[2] > half_chunk[1]):
            new_chunk[:, half_chunk[2]:, half_chunk[1]:,
                      :half_chunk[0]] = (
                          load_and_downscale_old_chunk(
                              z_idx * chunk_fetch_factor[2] + 1,
                              y_idx * chunk_fetch_factor[1] + 1,
                              x_idx * chunk_fetch_factor[0]))
        if new_chunk.shape[3] > half_chunk[0]:
            new_chunk[:, :half_chunk[2], :half_chunk[1],
                      half_chunk[0]:] = (
                          load_and_downscale_old_chunk(
                              z_idx * chunk_fetch_factor[2],
                              y_idx * chunk_fetch_factor[1],
                              x_idx * chunk_fetch_factor[0] + 1))
        if (new_chunk.shape[1] > half_chunk[2]
                and new_chunk.shape[3] > half_chunk[0]):
            new_chunk[:, half_chunk[2]:, :half_chunk[1],
                      half_chunk[0]:] = (
                          load_and_downscale_old_chunk(
                              z_idx * chunk_fetch_factor[2] + 1,
                              y_idx * chunk_fetch_factor[1],
                              x_idx * chunk_fetch_factor[0] + 1))
        if (new_chunk.shape[2] > half_chunk[1]
                and new_chunk.shape[3] > half_chunk[0]):
            new_chunk[:, :half_chunk[2], half_chunk[1]:,
                      half_chunk[0]:] = (
                          load_and_downscale_old_chunk(
                              z_idx * chunk_fetch_factor[2],
                              y_idx * chunk_fetch_factor[1] + 1,
                              x_idx * chunk_fetch_factor[0] + 1))
        if (new_chunk.shape[1] > half_chunk[2]
                and new_chunk.shape[2] > half_chunk[1]
                and new_chunk.shape[3] > half_chunk[0]):
            new_chunk[:, half_chunk[2]:, half_chunk[1]:,
                      half_chunk[0]:] = (
                          load_and_downscale_old_chunk(
                              z_idx * chunk_fetch_factor[2] + 1,
                              y_idx * chunk_fetch_factor[1] + 1,
                              x_idx * chunk_fetch_factor[0] + 1))

        chunk_writer.write_chunk(
            new_chunk.astype(dtype), new_key, new_chunk_coords
        )
