#! /usr/bin/env python3
#
# Copyright (c) 2016, 2017, 2018 Forschungszentrum Juelich GmbH
# Author: Yann Leprince <y.leprince@fz-juelich.de>
#
# This software is made available under the MIT licence, see LICENCE.txt.


import gzip
import json
import os
import os.path
import sys

import numpy as np
from tqdm import tqdm

RAW_CHUNK_PATTERN = "{key}/{0}-{1}/{2}-{3}/{4}-{5}"
RAW_CHUNK_PATTERN_FLAT = "{key}/{0}-{1}_{2}-{3}_{4}-{5}"


class StridingDownscaler:
    def check_factors(self, downscaling_factors):
        return True

    def downscale(self, chunk, downscaling_factors):
        return chunk[:,
                     ::downscaling_factors[2],
                     ::downscaling_factors[1],
                     ::downscaling_factors[0]
        ]


class AveragingDownscaler:
    def __init__(self, outside_value=None):
        if outside_value is None:
            self.padding_mode = "edge"
            self.pad_kwargs = {}
        else:
            self.padding_mode = "constant"
            self.pad_kwargs = {"constant_values": outside_value}

    def check_factors(self, downscaling_factors):
        return all(f in (1, 2) for f in downscaling_factors)

    def downscale(self, chunk, downscaling_factors):
        dtype = chunk.dtype
        chunk = chunk.astype(np.float32)  # unbounded type for arithmetic

        if downscaling_factors[2] == 2:
            if chunk.shape[1] % 2 != 0:
                chunk = np.pad(chunk, ((0, 0), (0, 1), (0, 0), (0, 0)),
                               self.padding_mode, **self.pad_kwargs)
            chunk = (chunk[:, ::2, :, :] + chunk[:, 1::2, :, :]) * 0.5

        if downscaling_factors[1] == 2:
            if chunk.shape[2] % 2 != 0:
                chunk = np.pad(chunk, ((0, 0), (0, 0), (0, 1), (0, 0)),
                               self.padding_mode, **self.pad_kwargs)
            chunk = (chunk[:, :, ::2, :] + chunk[:, :, 1::2, :]) * 0.5

        if downscaling_factors[0] == 2:
            if chunk.shape[3] % 2 != 0:
                chunk = np.pad(chunk, ((0, 0), (0, 0), (0, 0), (0, 1)),
                               self.padding_mode, **self.pad_kwargs)
            chunk = (chunk[:, :, :, ::2] + chunk[:, :, :, 1::2]) * 0.5

        return chunk.astype(dtype)


class ModeDownscaler:
    def check_factors(self, downscaling_factors):
        return True

    def downscale(self, chunk, downscaling_factors):
        # This could be optimized a lot (clever iteration with nditer, Cython)
        new_chunk = np.empty(
            (chunk.shape[0],
             (chunk.shape[1] - 1) // downscaling_factors[2] + 1,
             (chunk.shape[2] - 1) // downscaling_factors[1] + 1,
             (chunk.shape[3] - 1) // downscaling_factors[0] + 1),
            dtype=chunk.dtype
        )
        for t, z, y, x in np.ndindex(*new_chunk.shape):
            zd = z * downscaling_factors[2]
            yd = y * downscaling_factors[2]
            xd = x * downscaling_factors[2]
            block = chunk[t,
                          zd:(zd + downscaling_factors[2]),
                          yd:(yd + downscaling_factors[2]),
                          xd:(xd + downscaling_factors[2])
            ]

            labels, counts = np.unique(block.flat, return_counts=True)
            new_chunk[t, z, y, x] = labels[np.argsort(counts)[-1]]

        return new_chunk


def instantiate_downscaler(downscaling_method, outside_value):
    if downscaling_method == "average":
        return AveragingDownscaler(outside_value)
    elif downscaling_method == "mode":
        return ModeDownscaler()
    elif downscaling_method == "stride":
        return StridingDownscaler()


def create_next_scale(info, source_scale_index, downscaler,
                      flat_folder=False, compress=True):
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

    if flat_folder:
        chunk_pattern = RAW_CHUNK_PATTERN_FLAT
    else:
        chunk_pattern = RAW_CHUNK_PATTERN

    def load_and_downscale_old_chunk(z_idx, y_idx, x_idx):
        xmin = old_chunk_size[0] * x_idx
        xmax = min(old_chunk_size[0] * (x_idx + 1), old_size[0])
        ymin = old_chunk_size[1] * y_idx
        ymax = min(old_chunk_size[1] * (y_idx + 1), old_size[1])
        zmin = old_chunk_size[2] * z_idx
        zmax = min(old_chunk_size[2] * (z_idx + 1), old_size[2])
        chunk_filename = chunk_pattern.format(
            xmin, xmax, ymin, ymax, zmin, zmax, key=old_key)

        try:
            f = open(chunk_filename, "rb")
        except OSError:
            f = gzip.open(chunk_filename + ".gz", "rb")
        with f:
            chunk = np.frombuffer(f.read(), dtype=dtype).reshape(
                [num_channels, zmax - zmin, ymax - ymin, xmax - xmin])

        return downscaler.downscale(chunk, downscaling_factors)

    progress_bar = tqdm(
        total=(((new_size[0] - 1) // new_chunk_size[0] + 1)
               * ((new_size[1] - 1) // new_chunk_size[1] + 1)
               * ((new_size[2] - 1) // new_chunk_size[2] + 1)),
        desc="computing scale {}".format(new_key), unit="chunks", leave=True)
    for x_idx in range((new_size[0] - 1) // new_chunk_size[0] + 1):
        for y_idx in range((new_size[1] - 1) // new_chunk_size[1] + 1):
            for z_idx in range((new_size[2] - 1) // new_chunk_size[2] + 1):
                xmin = new_chunk_size[0] * x_idx
                xmax = min(new_chunk_size[0] * (x_idx + 1), new_size[0])
                ymin = new_chunk_size[1] * y_idx
                ymax = min(new_chunk_size[1] * (y_idx + 1), new_size[1])
                zmin = new_chunk_size[2] * z_idx
                zmax = min(new_chunk_size[2] * (z_idx + 1), new_size[2])
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

                new_chunk_name = chunk_pattern.format(
                    xmin, xmax, ymin, ymax, zmin, zmax, key=new_key)
                os.makedirs(os.path.dirname(new_chunk_name), exist_ok=True)
                if compress:
                    with gzip.open(new_chunk_name + ".gz", "wb") as f:
                        f.write(new_chunk.astype(dtype).tobytes())
                else:
                    with open(new_chunk_name, "wb") as f:
                        f.write(new_chunk.astype(dtype).tobytes())
                progress_bar.update()


def compute_scales(downscaling_method="average", outside_value=None,
                   flat_folder=False, compress=True):
    """Generate lower scales following an input info file"""
    downscaler = instantiate_downscaler(downscaling_method, outside_value)
    with open("info") as f:
        info = json.load(f)
    for i in range(len(info["scales"]) - 1):
        create_next_scale(info, i, downscaler,
                          flat_folder=flat_folder, compress=compress)


def parse_command_line(argv):
    """Parse the script's command line."""
    import argparse
    parser = argparse.ArgumentParser(
        description="""\
Create lower scales in Neuroglancer precomputed raw format

The list of scales is read from a file named "info" in the current directory.
""")
    parser.add_argument("--downscaling-method", default="average",
                        choices=("average", "mode", "stride"),
                        help='"average" is recommended for grey-level images, '
                        '"mode" for segmentation images. "stride" is fastest, '
                        'but provides no protection against aliasing '
                        'artefacts.')
    parser.add_argument("--outside-value", type=float, default=None,
                        help="padding value used by the 'average' downscaling "
                        "method for computing the voxels at the border. If "
                        "omitted, the volume is padded with its edge values.")
    parser.add_argument("--flat", action="store_true", dest="flat_folder",
                        help="Store all chunks for each resolution with a "
                        "flat layout, as neuroglancer expects. By default the "
                        "chunks are stored in sub-directories, which requires "
                        "a specially configured web server (see https://github"
                        ".com/HumanBrainProject/neuroglancer-docker). Do not "
                        "use this option for large images, or you risk "
                        "running into problems with directories containing "
                        "huge numbers of files.")
    parser.add_argument("--no-compression", action="store_false",
                        dest="compress",
                        help="Don't gzip the output.")
    args = parser.parse_args(argv[1:])
    return args


def main(argv):
    """The script's entry point."""
    args = parse_command_line(argv)
    return compute_scales(downscaling_method=args.downscaling_method,
                          outside_value=args.outside_value,
                          flat_folder=args.flat_folder,
                          compress=args.compress) or 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
