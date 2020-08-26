#! /usr/bin/env python3
#
# Copyright (c) 2016, 2017, Forschungszentrum Juelich GmbH
# Author: Yann Leprince <y.leprince@fz-juelich.de>
#
# This software is made available under the MIT licence, see LICENCE.txt.

from pathlib import Path
import sys

import numpy as np
import skimage.io
from tqdm import tqdm, trange

import neuroglancer_scripts.accessor
import neuroglancer_scripts.chunk_encoding
from neuroglancer_scripts.data_types import get_chunk_dtype_transformer
from neuroglancer_scripts import precomputed_io
from neuroglancer_scripts.utils import (permute, invert_permutation,
                                        readable_count)


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


def slices_to_raw_chunks(slice_filename_lists, dest_url, input_orientation,
                         options={}):
    """Convert a list of 2D slices to Neuroglancer pre-computed chunks.

    :param dict info: the JSON dictionary that describes the dataset for
        Neuroglancer. Only the information from the first scale is used.
    :param list slice_filename_lists: a list of lists of filenames. Files from
        each inner list are read as 2D images and concatenated along the third
        axis. Blocks from the outer list are concatenated along a fourth axis,
        representing the image channels.
    :param tuple input_axis_inversions: a 3-tuple in (column, row, slice)
        order. Each value must be 1 (preserve orientation along the axis) or -1
        (invert orientation along the axis).
    :param tuple input_axis_permutation: a 3-tuple in (column, row, slice)
      order. Each value is 0 for X (L-R axis), 1 for Y (A-P axis), 2 for Z (I-S
      axis).
    """
    accessor = neuroglancer_scripts.accessor.get_accessor_for_url(
        dest_url, options)
    pyramid_writer = precomputed_io.get_IO_for_existing_dataset(
        accessor, encoder_options=options
    )
    info = pyramid_writer.info

    assert len(info["scales"][0]["chunk_sizes"]) == 1  # more not implemented
    chunk_size = info["scales"][0]["chunk_sizes"][0]  # in RAS order (X, Y, Z)
    size = info["scales"][0]["size"]  # in RAS order (X, Y, Z)
    key = info["scales"][0]["key"]
    dtype = np.dtype(info["data_type"]).newbyteorder("<")
    num_channels = info["num_channels"]

    input_axis_permutation = tuple(AXIS_PERMUTATION_FOR_RAS[a]
                                   for a in input_orientation)
    input_axis_inversions = tuple(AXIS_INVERSION_FOR_RAS[a]
                                  for a in input_orientation)

    # Here x, y, and z refer to the data orientation in output chunks (which
    # should correspond to RAS+ anatomical axes). For the data orientation in
    # input slices the terms (column (index along image width), row (index
    # along image height), slice) are used.

    # permutation_to_input is a 3-tuple in RAS (X, Y, Z) order.
    # Each value is 0 column, 1 for row, 2 for slice.
    permutation_to_input = invert_permutation(input_axis_permutation)

    # input_size and input_chunk_size are in (column, row, slice) order.
    input_size = permute(size, input_axis_permutation)
    input_chunk_size = permute(chunk_size, input_axis_permutation)

    for filename_list in slice_filename_lists:
        if len(filename_list) != input_size[2]:
            raise ValueError("{} slices found where {} were expected"
                             .format(len(filename_list), input_size[2]))

    for slice_chunk_idx in trange((input_size[2] - 1)
                                  // input_chunk_size[2] + 1,
                                  desc="converting slice groups",
                                  leave=True, unit="slice groups"):
        first_slice_in_order = input_chunk_size[2] * slice_chunk_idx
        last_slice_in_order = min(input_chunk_size[2] * (slice_chunk_idx + 1),
                                  input_size[2])

        if input_axis_inversions[2] == -1:
            first_slice = input_size[2] - first_slice_in_order - 1
            last_slice = input_size[2] - last_slice_in_order - 1
        else:
            first_slice = first_slice_in_order
            last_slice = last_slice_in_order
        slice_slicing = np.s_[first_slice
                              : last_slice
                              : input_axis_inversions[2]]
        tqdm.write("Reading slices {0} to {1} ({2}B memory needed)... "
                   .format(first_slice, last_slice - input_axis_inversions[2],
                           readable_count(input_size[0]
                                          * input_size[1]
                                          * (last_slice_in_order
                                             - first_slice_in_order + 1)
                                          * num_channels
                                          * dtype.itemsize)))

        def load_z_stack(slice_filenames):
            # Loads the data in [slice, row, column] C-contiguous order
            block = skimage.io.concatenate_images(
                skimage.io.imread(str(filename))
                for filename in slice_filenames[slice_slicing]
            )
            assert block.shape[2] == input_size[0]  # check slice width
            assert block.shape[1] == input_size[1]  # check slice height
            if block.ndim == 4:
                # Scikit-image loads multi-channel (e.g. RGB) images in [slice,
                # row, column, channel] order, while Neuroglancer expects
                # channel to come first (in C-contiguous indexing).
                block = np.moveaxis(block, (3, 0, 1, 2), (0, 1, 2, 3))
            elif block.ndim == 3:
                block = block[np.newaxis, :, :, :]
            else:
                raise ValueError(
                    "block has unexpected dimensionality (ndim={})"
                    .format(block.ndim)
                )
            return block

        # Concatenate all channels from different directories
        block = np.concatenate([load_z_stack(filename_list)
                                for filename_list in slice_filename_lists],
                               axis=0)
        assert block.shape[0] == num_channels

        # Flip and permute axes to go from input (channel, slice, row, column)
        # to Neuroglancer (channel, Z, Y, X)
        block = block[:, :,
                      ::input_axis_inversions[1],
                      ::input_axis_inversions[0]]
        block = np.moveaxis(block, (3, 2, 1),
                            (3 - a for a in input_axis_permutation))
        # equivalent: np.transpose(block, axes=([0] + [3 - a for a in
        # reversed(invert_permutation(input_axis_permutation))]))

        chunk_dtype_transformer = get_chunk_dtype_transformer(
            block.dtype, dtype
        )

        progress_bar = tqdm(
            total=(((input_size[1] - 1) // input_chunk_size[1] + 1)
                   * ((input_size[0] - 1) // input_chunk_size[0] + 1)),
            desc="writing chunks", unit="chunks", leave=False)

        for row_chunk_idx in range((input_size[1] - 1)
                                   // input_chunk_size[1] + 1):
            row_slicing = np.s_[
                input_chunk_size[1] * row_chunk_idx
                : min(input_chunk_size[1] * (row_chunk_idx + 1),
                      input_size[1])
            ]
            for column_chunk_idx in range((input_size[0] - 1)
                                          // input_chunk_size[0] + 1):
                column_slicing = np.s_[
                    input_chunk_size[0] * column_chunk_idx
                    : min(input_chunk_size[0] * (column_chunk_idx + 1),
                          input_size[0])
                ]

                input_slicing = (column_slicing, row_slicing, np.s_[:])
                x_slicing, y_slicing, z_slicing = permute(
                    input_slicing, permutation_to_input)
                chunk = block[:, z_slicing, y_slicing, x_slicing]

                # This variable represents the coordinates with real slice
                # numbers, instead of within-block slice numbers.
                input_coords = (
                    (column_slicing.start, column_slicing.stop),
                    (row_slicing.start, row_slicing.stop),
                    (first_slice_in_order, last_slice_in_order)
                )
                x_coords, y_coords, z_coords = permute(
                    input_coords, permutation_to_input)
                assert chunk.size == ((x_coords[1] - x_coords[0])
                                      * (y_coords[1] - y_coords[0])
                                      * (z_coords[1] - z_coords[0])
                                      * num_channels)
                chunk_coords = (x_coords[0], x_coords[1],
                                y_coords[0], y_coords[1],
                                z_coords[0], z_coords[1])
                pyramid_writer.write_chunk(
                    chunk_dtype_transformer(chunk, preserve_input=False),
                    key, chunk_coords
                )
                progress_bar.update()
        # free up memory before reading next block (prevent doubled memory
        # usage)
        del block


def convert_slices_in_directory(slice_dirs, dest_url, input_orientation="RAS",
                                options={}):
    """Load slices from a directory and convert them to Neuroglancer chunks"""
    slice_filename_lists = [sorted(d.iterdir()) for d in slice_dirs]
    slices_to_raw_chunks(slice_filename_lists, dest_url, input_orientation,
                         options=options)


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
  (slices from the input directory are sorted in lexicographic order)

Each character can take one of six values, which represent the direction that
the axis **points to**:
- R for an axis that points towards anatomical Right
- L for an axis that points towards anatomical Left
- A for an axis that points towards Anterior
- P for an axis that points towards Posterior
- S for an axis that points towards Superior
- I for an axis that points towards Inferior

In the output chunks, the axes will be re-oriented accordingly to match RAS+
anatomical orientation.

A few examples:
- use “RIA” or “RIP” for coronal slices shown in neurological convention
- use “LIA” or “LIP” for coronal slices shown in radiological convention
- use “RPS” or “RPI” for axial slices shown in neurological convention
- use “LPS” or “LPI” for axial slices shown in neurological convention
""")
    parser.add_argument("slice_dirs", nargs="+", type=Path,
                        help="list of directories containing input slices,"
                        " slices from each directory will be loaded and"
                        " concatenated in lexicographic order, stacks from"
                        " different directories will be concatenated as"
                        " different channels")
    parser.add_argument("dest_url", help="directory/URL where the converted "
                        "dataset will be written")

    parser.add_argument("--input-orientation", default="RAS",
                        help="A 3-character code describing the anatomical"
                        " orientation of the input axes (see below) "
                        "[default: RAS]")

    # TODO add options for data conversion and scaling, like
    # volume_to_raw_chunks.py
    neuroglancer_scripts.accessor.add_argparse_options(parser)
    neuroglancer_scripts.chunk_encoding.add_argparse_options(parser,
                                                             allow_lossy=False)

    args = parser.parse_args(argv[1:])

    args.input_orientation = args.input_orientation.upper()
    if args.input_orientation not in POSSIBLE_AXIS_ORIENTATIONS:
        parser.error("input_orientation is invalid")
    return args


def main(argv=sys.argv):
    """The script's entry point."""
    import neuroglancer_scripts.utils
    neuroglancer_scripts.utils.init_logging_for_cmdline()
    args = parse_command_line(argv)
    return convert_slices_in_directory(args.slice_dirs,
                                       args.dest_url,
                                       args.input_orientation,
                                       options=vars(args)) or 0


if __name__ == "__main__":
    sys.exit(main())
