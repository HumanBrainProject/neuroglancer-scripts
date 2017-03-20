#! /usr/bin/env python3
#
# Copyright (c) 2016, 2017, Forschungszentrum Juelich GmbH
# Author: Yann Leprince <y.leprince@fz-juelich.de>
#
# This software is made available under the MIT licence, see LICENCE.txt.

import functools
import gzip
import json
import os
import os.path
import struct
import sys

import numpy as np


CHUNK_PATTERN = "{key}/{0}-{1}/{2}-{3}/{4}-{5}"


def make_seg_chunks(info, scale_index, raw_chunks_dir):
    """Convert chunks to compressed segmentation format for a specific scale"""

    with open(os.path.join(raw_chunks_dir, "info")) as f:
        input_info = json.load(f)
    input_dtype = np.dtype(input_info["data_type"]).newbyteorder("<")

    if info["data_type"] != "uint32" and info["data_type"] != "uint64":
        raise ValueError("compressed segmentation format can only encode"
                         " uint32 or uint64 data_type")
    dtype = np.dtype(info["data_type"]).newbyteorder("<")
    num_channels = info["num_channels"]
    scale_info = info["scales"][scale_index]
    key = scale_info["key"]
    size = scale_info["size"]
    assert scale_info["encoding"] == "compressed_segmentation"
    block_size = scale_info["compressed_segmentation_block_size"]

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
                    raw_chunk_filename = os.path.join(
                        raw_chunks_dir, CHUNK_PATTERN.format(
                        xmin, xmax, ymin, ymax, zmin, zmax, key=key))
                    shape = (num_channels,
                             zmax - zmin, ymax - ymin, xmax - xmin)
                    try:
                        f = open(raw_chunk_filename, "rb")
                    except OSError:
                        f = gzip.open(raw_chunk_filename + ".gz", "rb")
                    with f:
                        chunk = (np.frombuffer(f.read(), dtype=input_dtype)
                                 .reshape(shape).astype(dtype))

                    # Construct file in memory step by step
                    buf = bytearray(4 * num_channels)

                    for channel in range(num_channels):
                        # Write offset of the current channel into the header
                        assert len(buf) % 4 == 0
                        struct.pack_into("<I", buf, channel * 4, len(buf) // 4)

                        buf += compress_one_channel(chunk[channel, :, :, :],
                                                    block_size)

                    seg_chunk_filename = CHUNK_PATTERN.format(
                        xmin, xmax, ymin, ymax, zmin, zmax, key=key)
                    print("Writing", seg_chunk_filename)
                    os.makedirs(os.path.dirname(seg_chunk_filename),
                                exist_ok=True)
                    with gzip.open(seg_chunk_filename + ".gz", "wb") as f:
                        f.write(buf)


def compress_one_channel(chunk_channel, block_size):
    # Grid size (number of blocks of size in the chunk)
    gx = (chunk_channel.shape[2] - 1) // block_size[0] + 1
    gy = (chunk_channel.shape[1] - 1) // block_size[1] + 1
    gz = (chunk_channel.shape[0] - 1) // block_size[2] + 1
    stored_LUT_offsets = {}
    buf = bytearray(gx * gy * gz * 8)
    for z in range(gz):
        for y in range(gy):
            for x in range(gx):
                block = chunk_channel[
                    z * block_size[2]:(z + 1) * block_size[2],
                    y * block_size[1]:(y + 1) * block_size[1],
                    x * block_size[0]:(x + 1) * block_size[0]
                ]
                if block.shape != block_size:
                    block = pad_block(block, block_size)

                # TODO optimization: to improve additional compression (gzip),
                # sort the list of unique symbols by decreasing frequency using
                # return_counts=True so that low-value symbols are used more
                # often.
                (lookup_table, encoded_values) = np.unique(
                    block, return_inverse=True, return_counts=False)
                bits = number_of_encoding_bits(len(lookup_table))

                # Write look-up table to the buffer (or re-use another one)
                lut_bytes = lookup_table.astype(block.dtype).tobytes()
                if(lut_bytes in stored_LUT_offsets):
                    lookup_table_offset = stored_LUT_offsets[lut_bytes]
                else:
                    assert len(buf) % 4 == 0
                    lookup_table_offset = len(buf) // 4
                    buf += lut_bytes
                    stored_LUT_offsets[lut_bytes] = lookup_table_offset

                assert len(buf) % 4 == 0
                encoded_values_offset = len(buf) // 4
                buf += pack_encoded_values(encoded_values, bits)

                assert lookup_table_offset == (lookup_table_offset & 0xFFFFFF)
                struct.pack_into("<II", buf, 8 * (x + gx * (y + gy * z)),
                                 lookup_table_offset | (bits << 24),
                                 encoded_values_offset)
    return buf


def pad_block(block, block_size):
    """Pad a block to block_size with its most frequent value"""
    unique_vals, unique_counts = np.unique(block, return_counts=True)
    most_frequent_value = unique_vals[np.argmax(unique_counts)]
    return np.pad(block,
                  tuple((0, desired_size - actual_size)
                        for desired_size, actual_size
                        in zip(block_size, block.shape)),
                  mode="constant", constant_values=most_frequent_value)


def number_of_encoding_bits(elements):
    for bits in (0, 1, 2, 4, 8, 16, 32):
        if 2 ** bits >= elements:
            return bits
    raise ValueError("Too many elements!")

def pack_encoded_values(encoded_values, bits):
    # TODO optimize with np.packbits for bits == 1
    if bits == 0:
        return bytes()
    else:
        values_per_32bit = 32 // bits
        assert np.all(encoded_values == encoded_values & ((1 << bits) - 1))
        padded_values = np.empty(
            (values_per_32bit * (len(encoded_values) - 1)
             // values_per_32bit + 1),
            dtype="<I")
        padded_values[:len(encoded_values)] = encoded_values
        padded_values[len(encoded_values):] = 0
        packed_values = functools.reduce(
            np.bitwise_or,
            (padded_values[shift::values_per_32bit] << (shift * bits)
             for shift in range(values_per_32bit)))
        return packed_values.tobytes()


def convert_chunks_to_seg(raw_chunks_dir):
    """Convert precomputed chunks from raw to compressed segmentation format"""
    with open("info") as f:
        info = json.load(f)
    for scale_index in range(len(info["scales"])):
        make_seg_chunks(info, scale_index, raw_chunks_dir)


def parse_command_line(argv):
    """Parse the script's command line."""
    import argparse
    parser = argparse.ArgumentParser(
        description="""\
Convert Neuroglancer precomputed chunks from raw to compressed segmentation
 format

The list of scales is read from a file named "info" in the current directory.
""")
    parser.add_argument("raw_chunks_dir",
                        help="directory where the input raw chunks are found")
    args = parser.parse_args(argv[1:])
    return args


def main(argv):
    """The script's entry point."""
    args = parse_command_line(argv)
    return convert_chunks_to_seg(args.raw_chunks_dir) or 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
