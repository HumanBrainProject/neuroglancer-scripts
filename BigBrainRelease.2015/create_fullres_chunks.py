#! /usr/bin/env python3
#
# Copyright (c) 2016 Forschungszentrum Juelich GmbH
# Author: Yann Leprince <y.leprince@fz-juelich.de>
#
# This software is made available under the MIT licence, see LICENCE.txt.


import os
import os.path

import numpy as np

# This script saves the raw chunks in NumPy compressed format instead of the
# raw format used by Neuroglancer

slice_url_scheme = "/localdata/phd/BigBrain/bigbrain.loris.ca/BigBrainRelease.2015/2D_Final_Sections/Coronal/Png/Full_Resolution/pm{0:04d}o.png"
npz_chunk_scheme = "BigBrainRelease.2015/npz/{key}/{0:03d}/{1:03d}/{2:03d}.npz"
jpeg_chunk_scheme = "BigBrainRelease.2015/jpeg/{key}/{0}-{1}/{2}-{3}/{4}-{5}"

def create_fullres_chunks(full_scale_info):
    import skimage.io
    assert len(full_scale_info["chunk_sizes"]) == 1  # more not implemented
    chunk_size = full_scale_info["chunk_sizes"][0]
    size = full_scale_info["size"]

    # Process chunk_size[1] (e.g. 64) coronal slices at a time
    for coronal_chunk_idx in range((size[1] - 1) // chunk_size[1] + 1):
        coronal_chunk_start = coronal_chunk_idx * chunk_size[1]
        slice_numbers = np.r_[coronal_chunk_start + 1:
                              min(coronal_chunk_start + 65, size[1] + 1)]
        print("Now reading coronal slices {0} to {1}"
              .format(slice_numbers[0], slice_numbers[-1]))
        block = skimage.io.concatenate_images(
            skimage.io.imread(slice_url_scheme.format(slice_number))
            for slice_number in slice_numbers)
        coronal_slicing = np.s_[
            coronal_chunk_idx * chunk_size[1]:
            min((coronal_chunk_idx + 1) * chunk_size[1], size[0])]
        for axial_chunk_idx in range((size[2] - 1) // chunk_size[2] + 1):
            axial_slicing = np.s_[
                axial_chunk_idx * chunk_size[2]:
                min((axial_chunk_idx + 1) * chunk_size[2], size[2])]
            for sagittal_chunk_idx in range((size[0] - 1) // chunk_size[0] + 1):
                sagittal_slicing = np.s_[
                    sagittal_chunk_idx * chunk_size[0]:
                    min((sagittal_chunk_idx + 1) * chunk_size[0], size[0])]
                chunk = np.moveaxis(block[:, axial_slicing, sagittal_slicing],
                                    [2, 0, 1], [0, 1, 2])
                chunk_name = npz_chunk_scheme.format(sagittal_chunk_idx,
                                                     coronal_chunk_idx,
                                                     axial_chunk_idx,
                                                     key="full")
                assert chunk.shape[0] <= chunk_size[0]
                assert chunk.shape[1] <= chunk_size[1]
                assert chunk.shape[2] <= chunk_size[2]
                os.makedirs(os.path.dirname(chunk_name), exist_ok=True)
                print("Writing", chunk_name)
                with open(chunk_name, "wb") as f:
                    np.savez(f, chunk=chunk)
