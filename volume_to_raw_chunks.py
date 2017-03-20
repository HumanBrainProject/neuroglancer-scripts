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
import nibabel
import nibabel.orientations


RAW_CHUNK_PATTERN = "{key}/{0}-{1}/{2}-{3}/{4}-{5}"


def volume_to_raw_chunks(info, volume):
    assert len(info["scales"][0]["chunk_sizes"]) == 1  # more not implemented
    chunk_size = info["scales"][0]["chunk_sizes"][0]  # in order x, y, z
    size = info["scales"][0]["size"]  # in order x, y, z
    dtype = np.dtype(info["data_type"]).newbyteorder("<")
    num_channels = info["num_channels"]

    if volume.ndim < 4:
        volume = np.atleast_3d(volume)[:, :, :, np.newaxis]
    elif volume.ndim > 4:
        raise ValueError("Volumes with more than 4 dimensions not supported")

    # Volume given by nibabel are using Fortran indexing (X, Y, Z, T)
    assert volume.shape[:3] == tuple(size)
    assert volume.shape[3] == num_channels

    for x_chunk_idx in range((size[0] - 1) // chunk_size[0] + 1):
        x_slicing = np.s_[chunk_size[0] * x_chunk_idx:
                          min(chunk_size[0] * (x_chunk_idx + 1), size[0])]
        for y_chunk_idx in range((size[1] - 1) // chunk_size[1] + 1):
            y_slicing = np.s_[chunk_size[1] * y_chunk_idx:
                              min(chunk_size[1] * (y_chunk_idx + 1), size[1])]
            for z_chunk_idx in range((size[2] - 1) // chunk_size[2] + 1):
                z_slicing = np.s_[chunk_size[2] * z_chunk_idx:
                                  min(chunk_size[2] * (z_chunk_idx + 1), size[2])]
                chunk = np.moveaxis(volume[x_slicing, y_slicing, z_slicing, :],
                                    (0, 1, 2, 3), (3, 2, 1, 0))
                assert chunk.size == ((x_slicing.stop - x_slicing.start) *
                                      (y_slicing.stop - y_slicing.start) *
                                      (z_slicing.stop - z_slicing.start) *
                                      num_channels)

                chunk_name = RAW_CHUNK_PATTERN.format(
                    x_slicing.start, x_slicing.stop,
                    y_slicing.start, y_slicing.stop,
                    z_slicing.start, z_slicing.stop,
                    key=info["scales"][0]["key"])
                os.makedirs(os.path.dirname(chunk_name), exist_ok=True)
                with gzip.open(chunk_name + ".gz", "wb") as f:
                    f.write(chunk.astype(dtype).tobytes())


def volume_file_to_raw_chunks(volume_filename):
    """Convert from neuro-imaging formats to pre-computed raw chunks"""
    with open("info") as f:
        info = json.load(f)

    img = nibabel.load(volume_filename)
    affine = img.affine
    ornt = nibabel.orientations.io_orientation(affine)
    print("Detected input axis orientations {0}+"
          .format("".join(nibabel.orientations.ornt2axcodes(ornt))))
    new_affine = affine * nibabel.orientations.inv_ornt_aff(ornt, img.shape)
    input_voxel_sizes = nibabel.affines.voxel_sizes(affine)
    info_voxel_sizes = 0.000001 * np.asarray(info["scales"][0]["resolution"])
    print("Input voxel size is {0} mm".format(input_voxel_sizes))
    if not np.allclose(input_voxel_sizes, info_voxel_sizes):
        print("ERROR: voxel size is inconsistent with resolution in the info"
              " file({0} mm)".format(info_voxel_sizes))
        return 1

    sys.stderr.write("Loading volume... ")
    sys.stderr.flush()
    volume = nibabel.orientations.apply_orientation(img.get_data(), ornt)
    sys.stderr.write("done.\n")
    print("Loaded volume has data type {0}, chunks will be saved with {1}"
          .format(volume.dtype.name, info["data_type"]))

    sys.stderr.write("Writing chunks... ")
    sys.stderr.flush()
    volume_to_raw_chunks(info, volume)
    sys.stderr.write("done.\n")

    # This is the affine of the converted volume, print it at the end so it
    # does not get lost in scrolling
    print("Affine transformation of the converted volume:\n{0}"
          .format(new_affine))


def parse_command_line(argv):
    """Parse the script's command line."""
    import argparse
    parser = argparse.ArgumentParser(
        description="""\
Convert from neuro-imaging formats to Neuroglancer pre-computed raw chunks

The affine transformation on the input volume (as read by Nibabel) is to point
to a RAS+ oriented space. Chunks are saved in RAS+ order (X from left to Right,
Y from posterior to Anterior, Z from inferior to Superior).
""")
    parser.add_argument("volume_filename")
    args = parser.parse_args(argv[1:])
    return args


def main(argv):
    """The script's entry point."""
    args = parse_command_line(argv)
    return volume_file_to_raw_chunks(args.volume_filename) or 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
