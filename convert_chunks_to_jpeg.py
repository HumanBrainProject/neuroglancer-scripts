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
import PIL.Image


raw_chunk_pattern = "raw/{key}/{0}-{1}/{2}-{3}/{4}-{5}"
jpeg_chunk_pattern = "jpeg/{key}/{0}-{1}/{2}-{3}/{4}-{5}"


def make_jpeg_chunks(info, scale_index, jpeg_quality=95, slicing_plane="xy"):
    """Make JPEG chunks for a specific scale"""

    dtype = np.dtype(info["data_type"])
    if dtype.byteorder != "|":
        dtype.byteorder = "<"
    num_channels = info["num_channels"]
    if dtype != np.uint8:
        raise ValueError("JPEG compression is only possible for uint8 type")
    if num_channels != 1 and num_channels !=3:
        raise ValueError("JPEG compression is only possible for"
                         " images with 1 or 3 channels")
    scale_info = info["scales"][scale_index]
    key = scale_info["key"]
    size = scale_info["size"]

    for chunk_size in scale_info["chunk_sizes"]:
        for x_idx in range((size[0] - 1) // chunk_size[0] + 1):
            for y_idx in range((size[1] - 1) // chunk_size[1] + 1):
                for z_idx in range((size[2] - 1) // chunk_size[2] + 1):
                    xmin = chunk_size[0] * x_idx
                    xmax = min(chunk_size[0] * (x_idx + 1), size[0])
                    ymin = chunk_size[1] * y_idx
                    ymax = min(chunk_size[1] * (y_idx + 1), size[1])
                    zmin = chunk_size[2] * z_idx
                    zmax = min(chunk_size[2] * (z_idx + 1), size[2])
                    raw_chunk_filename = raw_chunk_pattern.format(
                        xmin, xmax, ymin, ymax, zmin, zmax, key=key)
                    shape = (num_channels,
                             zmax - zmin, ymax - ymin, xmax - xmin)
                    try:
                        f = open(raw_chunk_filename, "rb")
                    except OSError:
                        f = gzip.open(raw_chunk_filename + ".gz", "rb")
                    with f:
                        chunk = (np.frombuffer(f.read(), dtype=dtype)
                                 .reshape(shape))

                    if slicing_plane == "xy":
                        reshaped_chunk = chunk.reshape(
                            shape[0], shape[1] * shape[2], shape[3])
                    elif slicing_plane == "xz":
                        reshaped_chunk = chunk.reshape(
                            shape[0], shape[1], shape[2] * shape[3])
                    else:
                        raise RuntimeError()

                    if num_channels == 1:
                        reshaped_chunk = np.squeeze(reshaped_chunk, 0)
                    else:
                        # Channels (RGB) need to be along the last axis for PIL
                        reshaped_chunk = np.swapaxes(reshaped_chunk, 0, 3)

                    jpeg_chunk_filename = jpeg_chunk_pattern.format(
                        xmin, xmax, ymin, ymax, zmin, zmax, key=key)
                    img = PIL.Image.fromarray(reshaped_chunk)
                    print("Writing", jpeg_chunk_filename)
                    os.makedirs(os.path.dirname(jpeg_chunk_filename),
                                exist_ok=True)
                    img.save(jpeg_chunk_filename,
                             format="jpeg",
                             quality=jpeg_quality,
                             optimize=True,
                             progressive=True)

def convert_chunks_to_jpeg(jpeg_quality=95,
                           slicing_plane="xy"):
    """Convert Neuroglancer precomputed chunks from raw to jpeg format"""
    with open("info") as f:
        info = json.load(f)
    for scale_index in range(4, len(info["scales"]) - 1):
        make_jpeg_chunks(info, scale_index,
                         jpeg_quality=jpeg_quality,
                         slicing_plane=slicing_plane)


def parse_command_line(argv):
    """Parse the script's command line."""
    import argparse
    parser = argparse.ArgumentParser(
        description="""\
Convert Neuroglancer precomputed chunks from raw to jpeg format

The list of scales is read from a file named "info" in the current directory.
""")
    parser.add_argument("--jpeg-quality", type=int, default=95,
                        help="JPEG quality factor (0-95, values above 95"
                        " increase file size but provide little extra quality)")
    parser.add_argument("--slicing-plane", choices=("xy", "xz"), default="xy",
                        help="axis of planes that will be JPEG-compressed"
                        " (default: xy)")
    args = parser.parse_args(argv[1:])
    return args


def main(argv):
    """The script's entry point."""
    args = parse_command_line(argv)
    return convert_chunks_to_jpeg(jpeg_quality=args.jpeg_quality,
                                  slicing_plane=args.slicing_plane) or 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
