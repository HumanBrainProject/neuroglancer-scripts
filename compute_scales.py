#! /usr/bin/env python3
#
# Copyright (c) 2016, 2017, Forschungszentrum Juelich GmbH
# Author: Yann Leprince <y.leprince@fz-juelich.de>
#
# This software is made available under the MIT licence, see LICENCE.txt.


import gzip
import json
import os
import os.path
import sys

import numpy as np


RAW_CHUNK_PATTERN = "{key}/{0}-{1}/{2}-{3}/{4}-{5}"


def create_next_scale(info, source_scale_index, outside_value=0):
    # Key is the resolution in micrometres
    old_scale_info = info["scales"][source_scale_index]
    new_scale_info = info["scales"][source_scale_index + 1]
    old_chunk_size = old_scale_info["chunk_sizes"][0]
    new_chunk_size = new_scale_info["chunk_sizes"][0]
    old_key = old_scale_info["key"]
    new_key = new_scale_info["key"]
    old_size = old_scale_info["size"]
    new_size = new_scale_info["size"]
    dtype = np.dtype(info["data_type"])
    if dtype.byteorder != "|":
        dtype.byteorder = "<"
    num_channels = info["num_channels"]
    downscaling_factors = [(os + 1) // ns
                           for os, ns in zip(old_size, new_size)]
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
        chunk_filename = RAW_CHUNK_PATTERN.format(
            xmin, xmax, ymin, ymax, zmin, zmax, key=old_key)

        try:
            f = open(chunk_filename, "rb")
        except OSError:
            f = gzip.open(chunk_filename + ".gz", "rb")
        with f:
            chunk = np.frombuffer(f.read(), dtype=dtype).reshape(
                [num_channels, zmax - zmin, ymax - ymin, xmax - xmin])
        chunk = chunk.astype(np.float32)  # unbounded type for arithmetic

        if downscaling_factors[2] == 2:
            if chunk.shape[1] % 2 != 0:
                chunk = np.pad(chunk, ((0, 0), (0, 1), (0, 0), (0, 0)),
                               "constant", constant_values=outside_value)
            chunk = (chunk[:, ::2, :, :] + chunk[:, 1::2, :, :]) * 0.5

        if downscaling_factors[1] == 2:
            if chunk.shape[2] % 2 != 0:
                chunk = np.pad(chunk, ((0, 0), (0, 0), (0, 1), (0, 0)),
                               "constant", constant_values=outside_value)
            chunk = (chunk[:, :, ::2, :] + chunk[:, :, 1::2, :]) * 0.5

        if downscaling_factors[0] == 2:
            if chunk.shape[3] % 2 != 0:
                chunk = np.pad(chunk, ((0, 0), (0, 0), (0, 0), (0, 1)),
                               "constant", constant_values=outside_value)
            chunk = (chunk[:, :, :, ::2] + chunk[:, :, :, 1::2]) * 0.5

        return chunk.astype(dtype)

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
                              half_chunk[0]:] = (load_and_downscale_old_chunk(
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

                new_chunk_name = RAW_CHUNK_PATTERN.format(
                    xmin, xmax, ymin, ymax, zmin, zmax, key=new_key)
                print("Writing", new_chunk_name)
                os.makedirs(os.path.dirname(new_chunk_name), exist_ok=True)
                with gzip.open(new_chunk_name + ".gz", "wb") as f:
                    f.write(new_chunk.astype(dtype).tobytes())


def compute_scales(outside_value):
    """Generate lower scales following an input info file"""
    with open("info") as f:
        info = json.load(f)
    for i in range(len(info["scales"]) - 1):
        create_next_scale(info, i, outside_value=outside_value)


def parse_command_line(argv):
    """Parse the script's command line."""
    import argparse
    parser = argparse.ArgumentParser(
        description="""\
Create lower scales in Neuroglancer precomputed raw format

The list of scales is read from a file named "info" in the current directory.
""")
    parser.add_argument("--outside-value", default=0)
    args = parser.parse_args(argv[1:])
    return args


def main(argv):
    """The script's entry point."""
    args = parse_command_line(argv)
    return compute_scales(outside_value=args.outside_value) or 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
