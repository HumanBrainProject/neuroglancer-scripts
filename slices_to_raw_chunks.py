#! /usr/bin/env python3
#
# Copyright (c) 2016, 2017, Forschungszentrum Juelich GmbH
# Author: Yann Leprince <y.leprince@fz-juelich.de>
#
# This software is made available under the MIT licence, see LICENCE.txt.

import json
import os
import os.path
import sys

import numpy as np
import skimage.io


slice_url_pattern = "completed_slices/B20_stn_l_sebastian_{0:04d}.tif"
raw_chunk_pattern = "raw/{key}/{0}-{1}/{2}-{3}/{4}-{5}"

def get_slice_filename(slice_number):
    return slice_url_pattern.format(slice_number)

def slices_to_raw_chunks(info):
    assert len(info["scales"][0]["chunk_sizes"]) == 1  # more not implemented
    chunk_size = info["scales"][0]["chunk_sizes"][0]
    size = info["scales"][0]["size"]
    dtype = np.dtype(info["data_type"])
    if dtype.byteorder != "|":
        dtype.byteorder = "<"

    for z_chunk_idx in range((size[2] - 1) // chunk_size[2] + 1):
        z_slicing = np.s_[chunk_size[2] * z_chunk_idx:
                          min(chunk_size[2] * (z_chunk_idx + 1), size[2])]

        print("Reading y slices {0} to {1}... "
              .format(z_slicing.start, z_slicing.stop - 1), end="")
        # Loads the data in [Z, Y, X] C-contiguous order ([slice, row, column])
        block = skimage.io.concatenate_images(
            skimage.io.imread(get_slice_filename(slice_number))
            for slice_number in np.r_[z_slicing])
        if block.ndim == 4:
            # Scikit-image loads multi-channel (e.g. RGB) images in [Z, Y, X,
            # channel] order, while Neuroglancer expects [channel, Z, Y, X] (in
            # C-contiguous indexing).
            assert block.shape[3] == info["num_channels"]
            block = numpy.swapaxes(block, 0, 3)
        elif block.ndim == 3:
            block = block[np.newaxis, :, :, :]
        else:
            raise ValueError("block has unknown dimensionality")

        print("writing chunks...")
        for y_chunk_idx in range((size[1] - 1) // chunk_size[1] + 1):
            y_slicing = np.s_[chunk_size[1] * y_chunk_idx:
                              min(chunk_size[1] * (y_chunk_idx + 1), size[1])]
            for x_chunk_idx in range((size[0] - 1) // chunk_size[0] + 1):
                x_slicing = np.s_[chunk_size[0] * x_chunk_idx:
                                  min(chunk_size[0] * (x_chunk_idx + 1), size[0])]
                chunk = block[:, :, y_slicing, x_slicing]
                chunk_name = raw_chunk_pattern.format(
                    x_slicing.start, x_slicing.stop,
                    y_slicing.start, y_slicing.stop,
                    z_slicing.start, z_slicing.stop,
                    key=info["scales"][0]["key"])
                os.makedirs(os.path.dirname(chunk_name), exist_ok=True)
                with open(chunk_name, "wb") as f:
                    f.write(chunk.astype(dtype).tobytes())


if __name__ == "__main__":
    with open("info") as f:
        info = json.load(f)
    slices_to_raw_chunks(info)
