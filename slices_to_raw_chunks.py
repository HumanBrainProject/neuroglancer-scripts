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
import skimage.io

# Generated with the following Python expression:
# >>> from itertools import *
# >>> ["".join(l) for t in product("LR", "AP", "IS") for l in permutations(t)]
POSSIBLE_AXIS_ORIENTATIONS = [
    "LAI", "LIA", "ALI", "AIL", "ILA", "IAL",
    "LAS", "LSA", "ALS", "ASL", "SLA", "SAL",
    "LPI", "LIP", "PLI", "PIL", "ILP", "IPL",
    "LPS", "LSP", "PLS", "PSL", "SLP", "SPL",
    "RAI", "RIA", "ARI", "AIR", "IRA", "IAR",
    "RAS", "RSA", "ARS", "ASR", "SRA", "SAR",
    "RPI", "RIP", "PRI", "PIR", "IRP", "IPR",
    "RPS", "RSP", "PRS", "PSR", "SRP", "SPR"
]

AXIS_PERMUTATION_FOR_RAS = {
    "R": 0,
    "L": 0,
    "A": 1,
    "P": 1,
    "S": 2,
    "I": 2
}

AXIS_INVERSION_FOR_RAS = {
    "R": 1,
    "A": 1,
    "S": 1,
    "L": -1,
    "P": -1,
    "I": -1
}

RAW_CHUNK_PATTERN = "{key}/{0}-{1}/{2}-{3}/{4}-{5}"


def permute(seq, p):
    """Permute the elements of seq according to the permutation p"""
    return tuple(seq[i] for i in p)

def invert_permutation(p):
    """The argument p is assumed to be some permutation of 0, 1, ..., len(p)-1.
    Returns an array s, where s[i] gives the index of i in p.
    """
    p = np.asarray(p)
    s = np.empty(p.size, p.dtype)
    s[p] = np.arange(p.size)
    return s


def slices_to_raw_chunks(info, slice_filename_lists,
                         input_axis_inversions, input_axis_permutation):
    assert len(info["scales"][0]["chunk_sizes"]) == 1  # more not implemented
    chunk_size = info["scales"][0]["chunk_sizes"][0]  # in order x, y, z
    size = info["scales"][0]["size"]  # in order x, y, z
    dtype = np.dtype(info["data_type"]).newbyteorder("<")
    num_channels = info["num_channels"]

    # Here x, y, and z refer to the data orientation in output chunks (which
    # should correspond to RAS+ anatomical axes). For the data orientation in
    # input slices the terms (column, row, slice) are used.

    permutation_to_input = invert_permutation(input_axis_permutation)

    for l in slice_filename_lists:
        assert len(l) == size[input_axis_permutation[2]]
    input_size = permute(size, permutation_to_input)
    input_chunk_size = permute(chunk_size, permutation_to_input)

    for slice_chunk_idx in range((input_size[2] - 1)
                                 // input_chunk_size[2] + 1):
        first_slice_in_order = input_chunk_size[2] * slice_chunk_idx
        last_slice_in_order = min(input_chunk_size[2] * (slice_chunk_idx + 1),
                                  input_size[2])

        if input_axis_permutation[2] == -1:
            first_slice = input_size[2] - first_slice_in_order - 1
            last_slice = input_size[2] - last_slice_in_order - 1
        else:
            first_slice = first_slice_in_order
            last_slice = last_slice_in_order
        slice_slicing = np.s_[first_slice
                              :last_slice
                              :input_axis_permutation[2]]
        print("Reading slices {0} to {1}... "
              .format(first_slice, last_slice - 1), end="")
        sys.stdout.flush()

        def load_z_stack(slice_filenames):
            # Loads the data in [slice, row, column] C-contiguous order
            block = skimage.io.concatenate_images(
                map(skimage.io.imread, slice_filenames[slice_slicing]))
            assert block.shape[2] == input_size[0]  # check slice width
            assert block.shape[1] == input_size[1]  # check slice height
            if block.ndim == 4:
                # Scikit-image loads multi-channel (e.g. RGB) images in [slice,
                # row, column, channel] order, while Neuroglancer expects channel
                # to come first (in C-contiguous indexing).
                block = np.swapaxes(block, 0, 3)
            elif block.ndim == 3:
                block = block[np.newaxis, :, :, :]
            else:
                raise ValueError("block has unexpected dimensionality (ndim={})"
                                 .format(block.ndim))
            return block

        block = np.concatenate([load_z_stack(l) for l in slice_filename_lists],
                               axis=0)
        assert block.shape[0] == num_channels

        # Flip and permute axes to go from input (channel, slice, row, column)
        # to Neuroglancer (channel, Z, Y, X)
        block = block[:, :,
                      ::input_axis_inversions[1],
                      ::input_axis_inversions[0]]
        block = np.moveaxis(block, (3, 2, 1),
                            (3 - a for a in input_axis_permutation))

        print("writing chunks...")
        for row_chunk_idx in range((input_size[1] - 1)
                                   // input_chunk_size[1] + 1):
            row_slicing = np.s_[
                input_chunk_size[1] * row_chunk_idx
                :min(input_chunk_size[1] * (row_chunk_idx + 1), input_size[1])]
            for column_chunk_idx in range((input_size[0] - 1)
                                          // input_chunk_size[0] + 1):
                column_slicing = np.s_[
                    input_chunk_size[0] * column_chunk_idx
                    :min(input_chunk_size[0] * (column_chunk_idx + 1),
                         input_size[0])]

                input_slicing = (column_slicing, row_slicing, np.s_[:])
                x_slicing , y_slicing, z_slicing = permute(
                    input_slicing, input_axis_permutation)
                chunk = block[:, z_slicing, y_slicing, x_slicing]

                # This variable represents the coordinates with real slice
                # numbers, instead of within-block slice numbers.
                input_coords = (
                    (column_slicing.start, column_slicing.stop),
                    (row_slicing.start, row_slicing.stop),
                    (first_slice_in_order, last_slice_in_order)
                )
                x_coords , y_coords, z_coords = permute(
                    input_coords, input_axis_permutation)
                assert chunk.size == ((x_coords[1] - x_coords[0]) *
                                      (y_coords[1] - y_coords[0]) *
                                      (z_coords[1] - z_coords[0]) *
                                      num_channels)

                chunk_name = RAW_CHUNK_PATTERN.format(
                    x_coords[0], x_coords[1],
                    y_coords[0], y_coords[1],
                    z_coords[0], z_coords[1],
                    key=info["scales"][0]["key"])
                os.makedirs(os.path.dirname(chunk_name), exist_ok=True)
                with gzip.open(chunk_name + ".gz", "wb") as f:
                    f.write(chunk.astype(dtype).tobytes())


def convert_slices_in_directory(slice_dirs, input_orientation):
    """Load slices from a directory and convert them to Neuroglancer chunks"""
    with open("info") as f:
        info = json.load(f)

    input_axis_permutation = tuple(AXIS_PERMUTATION_FOR_RAS[l]
                                   for l in input_orientation)
    input_axis_inversions = tuple(AXIS_INVERSION_FOR_RAS[l]
                                  for l in input_orientation)

    slice_filename_lists = [[os.path.join(d, filename)
                             for filename in sorted(os.listdir(d))]
                            for d in slice_dirs]
    slices_to_raw_chunks(info, slice_filename_lists,
                         input_axis_inversions, input_axis_permutation)

def parse_command_line(argv):
    """Parse the script's command line."""
    import argparse
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="""\
Convert a directory of 2D slices into a 3D volume in Neuroglancer chunk format

The list of scales is read from a file named "info" in the current directory.
This "info" file can be generated with generate_scales_info.py. """,
        epilog="""\
The orientation of the input axes must be specified using a 3-character code.
Each character represents the anatomical direction of an input axis:
- The first character represents the direction along rows of the input slices,
  i.e. the left-to-right on-screen axis when the image is displayed.
- The second character represents the direction along columns in the input
  slices, i.e. the top-to-bottom on-screen axis when the image is displayed.
- The third character represents the direction along increasing slice numbers
  (slices from the input directory are sorted in lexicographical order)

Each character can take one of six values, which represent the direction that
the axis **points to**:
- R for an axis that points towards anatomical Right
- L for an axis that points towards anatomical Left
- A for an axis that points towards Anterior
- P for an axis that points towards Posterior
- S for an axis that points towards Superior
- I for an axis that points towards Inferior

A few examples:
- use “RIA” or “RIP” for coronal slices shown in neurological convention
- use “LIA” or “LIP” for coronal slices shown in radiological convention
- use “RPS” or “RPI” for axial slices shown in neurological convention
- use “LPS” or “LPI” for axial slices shown in neurological convention
""")
    parser.add_argument("slice_dirs", nargs="+",
                        help="list of directories containing input slices,"
                        " slices from each directory will be loaded and"
                        " concatenated in lexicographic order, stacks from"
                        " different directories will be concatenated as"
                        " different channels")
    parser.add_argument("input_orientation",
                        help="A 3-character code describe the anatomical"
                        " orientation of the input axes (see above)")
    args = parser.parse_args(argv[1:])
    args.input_orientation = args.input_orientation.upper()
    if args.input_orientation not in POSSIBLE_AXIS_ORIENTATIONS:
        parser.error("input_orientation is invalid")
    return args


def main(argv):
    """The script's entry point."""
    args = parse_command_line(argv)
    return convert_slices_in_directory(args.slice_dirs,
                                       args.input_orientation) or 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
