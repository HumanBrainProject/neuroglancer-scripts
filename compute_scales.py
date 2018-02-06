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
import neuroglancer_scripts.downscaling
import neuroglancer_scripts.pyramid_io


def create_next_scale(info, source_scale_index, downscaler,
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
    if new_size != [(os - 1) // ds + 1
                    for os, ds in zip(old_size, downscaling_factors)]:
        raise ValueError("Unsupported downscaling factor between scales "
                         "{} and {} (only 1 and 2 are supported)"
                         .format(old_key, new_key))

    downscaler.check_factors(downscaling_factors)

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

    chunk_range = ((new_size[0] - 1) // new_chunk_size[0] + 1,
                   (new_size[1] - 1) // new_chunk_size[1] + 1,
                   (new_size[2] - 1) // new_chunk_size[2] + 1)
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


def compute_scales(work_dir=".", downscaling_method="average", options={}):
    """Generate lower scales following an input info file"""
    accessor = neuroglancer_scripts.accessor.get_accessor_for_url(
        work_dir, options)
    info = accessor.fetch_info()
    pyramid_io = neuroglancer_scripts.pyramid_io.PrecomputedPyramidIo(
        info, accessor, encoder_params=options)
    downscaler = neuroglancer_scripts.downscaling.get_downscaler(
        downscaling_method, options)
    for i in range(len(info["scales"]) - 1):
        create_next_scale(info, i, downscaler, pyramid_io, pyramid_io)


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


def main(argv):
    """The script's entry point."""
    args = parse_command_line(argv)
    return compute_scales(args.work_dir,
                          args.downscaling_method,
                          options=vars(args)) or 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
