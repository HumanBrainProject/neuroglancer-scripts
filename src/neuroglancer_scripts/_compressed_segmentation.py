# Copyright (c) 2016, 2017, 2018 Forschungszentrum Juelich GmbH
# Author: Yann Leprince <y.leprince@fz-juelich.de>
#
# This software is made available under the MIT licence, see LICENCE.txt.

import functools
import itertools
import struct

import numpy as np

from neuroglancer_scripts.chunk_encoding import InvalidFormatError
from neuroglancer_scripts.utils import ceil_div


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
    raise AssertionError("Too many elements")


COMPRESSED_SEGMENTATION_DATA_TYPES = (
    np.dtype(np.uint32).newbyteorder("<"),
    np.dtype(np.uint64).newbyteorder("<"),
)


def encode_chunk(chunk, block_size):
    # Construct file in memory step by step
    num_channels = chunk.shape[0]
    buf = bytearray(4 * num_channels)

    assert chunk.dtype in COMPRESSED_SEGMENTATION_DATA_TYPES

    for channel in range(num_channels):
        # Write offset of the current channel into the header
        assert len(buf) % 4 == 0
        struct.pack_into("<I", buf, channel * 4, len(buf) // 4)

        buf += _encode_channel(
            chunk[channel, :, :, :], block_size)
    return buf


def _encode_channel(chunk_channel, block_size):
    block_size = tuple(block_size)
    # Grid size (number of blocks in the chunk)
    gx = ceil_div(chunk_channel.shape[2], block_size[0])
    gy = ceil_div(chunk_channel.shape[1], block_size[1])
    gz = ceil_div(chunk_channel.shape[0], block_size[2])
    stored_lut_offsets = {}
    buf = bytearray(gx * gy * gz * 8)
    for z, y, x in np.ndindex((gz, gy, gx)):
        block = chunk_channel[
            z*block_size[2] : (z+1)*block_size[2],
            y*block_size[1] : (y+1)*block_size[1],
            x*block_size[0] : (x+1)*block_size[0]
        ]
        if block.shape != block_size:
            block = pad_block(block, block_size)

        # TODO optimization: to improve additional compression (gzip), sort the
        # list of unique symbols by decreasing frequency using
        # return_counts=True so that low-value symbols are used more often.
        # Alternatively, sort by label value to improve sharing of lookup
        # tables…
        (lookup_table, encoded_values) = np.unique(
            block, return_inverse=True, return_counts=False)
        bits = number_of_encoding_bits(len(lookup_table))
        encoded_values = encoded_values.ravel()  # Numpy 2.0 compat

        # Write look-up table to the buffer (or re-use another one)
        lut_bytes = lookup_table.astype(block.dtype).tobytes()
        if lut_bytes in stored_lut_offsets:
            lookup_table_offset = stored_lut_offsets[lut_bytes]
        else:
            assert len(buf) % 4 == 0
            lookup_table_offset = len(buf) // 4
            buf += lut_bytes
            stored_lut_offsets[lut_bytes] = lookup_table_offset

        assert len(buf) % 4 == 0
        encoded_values_offset = len(buf) // 4
        buf += _pack_encoded_values(encoded_values, bits)

        assert lookup_table_offset == (lookup_table_offset & 0xFFFFFF)
        struct.pack_into("<II", buf, 8 * (x + gx * (y + gy * z)),
                         lookup_table_offset | (bits << 24),
                         encoded_values_offset)
    return buf


def _pack_encoded_values(encoded_values, bits):
    if bits == 0:
        return b""
    else:
        assert 32 % bits == 0
        assert np.array_equal(encoded_values,
                              encoded_values & ((1 << bits) - 1))
        values_per_32bit = 32 // bits
        padded_values = np.pad(encoded_values.astype("<I", casting="unsafe"),
                               [(0, -len(encoded_values) % values_per_32bit)],
                               mode="constant", constant_values=0)
        assert len(padded_values) % values_per_32bit == 0
        packed_values = functools.reduce(
            np.bitwise_or,
            (padded_values[shift::values_per_32bit] << (shift * bits)
             for shift in range(values_per_32bit)))
        return packed_values.tobytes()


def decode_chunk_into(chunk, buf, block_size):
    num_channels = chunk.shape[0]
    # Grid size (number of blocks in the chunk)
    gx = ceil_div(chunk.shape[3], block_size[0])
    gy = ceil_div(chunk.shape[2], block_size[1])
    gz = ceil_div(chunk.shape[1], block_size[2])

    if len(buf) < num_channels * (4 + 8 * gx * gy * gz):
        raise InvalidFormatError("compressed_segmentation file too short")

    channel_offsets = [
        4 * ret[0]
        for ret in struct.iter_unpack("<I", buf[:4*num_channels])
    ]
    for channel, (offset, next_offset) in enumerate(
            itertools.zip_longest(channel_offsets,
                                  channel_offsets[1:])):
        # next_offset will be None for the last channel
        if offset + 8 * gx * gy * gz > len(buf):
            raise InvalidFormatError("compressed_segmentation channel offset "
                                     "is too large (truncated file?)")
        _decode_channel_into(
            chunk, channel, buf[offset:next_offset], block_size
        )

    return chunk


def _decode_channel_into(chunk, channel, buf, block_size):
    # Grid size (number of blocks in the chunk)
    gx = ceil_div(chunk.shape[3], block_size[0])
    gy = ceil_div(chunk.shape[2], block_size[1])
    gz = ceil_div(chunk.shape[1], block_size[2])
    block_num_elem = block_size[0] * block_size[1] * block_size[2]
    for z, y, x in np.ndindex((gz, gy, gx)):
        # Read the block header
        res = struct.unpack_from("<II", buf, 8 * (x + gx * (y + gy * z)))
        lookup_table_offset = 4 * (res[0] & 0x00FFFFFF)
        bits = res[0] >> 24
        if bits not in (0, 1, 2, 4, 8, 16, 32):
            raise InvalidFormatError("Invalid number of encoding bits for "
                                     f"compressed_segmentation block ({bits})"
                                     )
        encoded_values_offset = 4 * res[1]
        lookup_table_past_end = lookup_table_offset + chunk.itemsize * min(
            (2 ** bits),
            ((len(buf) - lookup_table_offset) // chunk.itemsize)
        )
        lookup_table = np.frombuffer(
            buf[lookup_table_offset:lookup_table_past_end], dtype=chunk.dtype)
        if bits == 0:
            block = np.empty(block_size, dtype=chunk.dtype)
            try:
                block[...] = lookup_table[0]
            except IndexError as exc:
                raise InvalidFormatError(
                    "Invalid compressed_segmentation data: indexing out of "
                    "the lookup table") from exc
        else:
            values_per_32bit = 32 // bits
            encoded_values_end = encoded_values_offset + 4 * (
                ceil_div(block_num_elem, values_per_32bit)
            )
            if encoded_values_end > len(buf):
                raise InvalidFormatError(
                    "Invalid compressed_segmentation data: file too short, "
                    "insufficient room for encoded values"
                )
            packed_values = np.frombuffer(buf[encoded_values_offset:
                                              encoded_values_end], dtype="<I")
            encoded_values = _unpack_encoded_values(packed_values, bits,
                                                    block_num_elem)
            # Apply the lookup table
            try:
                decoded_values = lookup_table[encoded_values]
            except IndexError as exc:
                raise InvalidFormatError(
                    "Invalid compressed_segmentation data: indexing out of "
                    "the lookup table") from exc
            block = decoded_values.reshape((block_size[2],
                                            block_size[1],
                                            block_size[0]))

        # Remove padding
        zmax = min(block_size[2], chunk.shape[1] - z * block_size[2])
        ymax = min(block_size[1], chunk.shape[2] - y * block_size[1])
        xmax = min(block_size[0], chunk.shape[3] - x * block_size[0])
        chunk[
            channel,
            z*block_size[2] : (z+1)*block_size[2],
            y*block_size[1] : (y+1)*block_size[1],
            x*block_size[0] : (x+1)*block_size[0]
        ] = block[:zmax, :ymax, :xmax]


def _unpack_encoded_values(packed_values, bits, num_values):
    assert bits > 0
    assert 32 % bits == 0
    bitmask = (1 << bits) - 1
    values_per_32bit = 32 // bits
    padded_values = np.empty(
        values_per_32bit * ceil_div(num_values, values_per_32bit),
        dtype="I")
    for shift in range(values_per_32bit):
        padded_values[shift::values_per_32bit] = (
            (packed_values >> (shift * bits)) & bitmask)
    return padded_values[:num_values]
