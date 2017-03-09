#! /usr/bin/env python3
#
# Copyright (c) 2016 Forschungszentrum Juelich GmbH
# Author: Yann Leprince <y.leprince@fz-juelich.de>
#
# This software is made available under the MIT licence, see LICENCE.txt.


import copy
import json
import math
import os
import os.path

import numpy as np


def create_info_json():
    with open("info_fullres.json") as f:
        info = json.load(f)

    assert len(info["scales"]) == 1
    full_scale_info = info["scales"][0]

    full_resolution = full_scale_info["resolution"]
    best_axis_resolution = min(full_resolution)
    axis_factor_offsets = [
        int(round(math.log2(axis_res / best_axis_resolution)))
        for axis_res in full_resolution]

    def downscale_info(log2_factor):
        factors = [2 ** max(0, log2_factor - axis_factor_offset)
                   for axis_factor_offset in axis_factor_offsets]
        scale_info = copy.deepcopy(full_scale_info)
        scale_info["resolution"] = [
            res * axis_factor for res, axis_factor in
            zip(full_scale_info["resolution"], factors)]
        scale_info["size"] = [
            (sz - 1) // axis_factor + 1 for sz, axis_factor in
            zip(full_scale_info["size"], factors)]
        # Key is the resolution in micrometres
        scale_info["key"] = str(round(min(res / 1000 for res in
                                          scale_info["resolution"]))) + "um"
        return scale_info

    assert len(full_scale_info["chunk_sizes"]) == 1  # more not implemented
    max_downscale_factor = (
        max(math.ceil(math.log2(a / b)) - c
            for a, b, c in zip(full_scale_info["size"],
                               full_scale_info["chunk_sizes"][0],
                               axis_factor_offsets)))
    info["scales"] = [downscale_info(i)
                      for i in range(max_downscale_factor)]

    with open("info", "w") as f:
        json.dump(info, f)

def read_info_json():
    with open("info") as f:
        info = json.load(f)
    return info



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

outside_greylevel = 255

def create_next_scale(scale_info):
    new_scale_info = copy.deepcopy(scale_info)
    new_scale_info["resolution"] = [res * 2 for res in
                                     scale_info["resolution"]]
    new_scale_info["size"] = [(sz - 1) // 2 + 1 for sz in
                               scale_info["size"]]
    # Key is the resolution in micrometres
    new_scale_info["key"] = str(round(min(res / 1000 for res in
                                          new_scale_info["resolution"]))) + "um"
    chunk_size = scale_info["chunk_sizes"][0]
    half_chunk = [sz // 2 for sz in chunk_size]
    old_key = scale_info["key"]
    new_key = new_scale_info["key"]
    old_size = scale_info["size"]
    new_size = new_scale_info["size"]

    def load_and_downscale_old_chunk(sag_idx, cor_idx, ax_idx):
        chunk_filename = npz_chunk_scheme.format(sag_idx, cor_idx, ax_idx,
                                                 key=old_key)
        old_chunk = np.load(chunk_filename)["chunk"]
        if old_chunk.shape[0] % 2 != 0:
            old_chunk = np.pad(old_chunk, ((0, 1), (0, 0), (0, 0)),
                               "constant", constant_values=outside_greylevel)
        if old_chunk.shape[1] % 2 != 0:
            old_chunk = np.pad(old_chunk, ((0, 0), (0, 1), (0, 0)),
                               "constant", constant_values=outside_greylevel)
        if old_chunk.shape[2] % 2 != 0:
            old_chunk = np.pad(old_chunk, ((0, 0), (0, 0), (0, 1)),
                               "constant", constant_values=outside_greylevel)

        assert old_chunk.shape[0] % 2 == 0
        assert old_chunk.shape[1] % 2 == 0
        assert old_chunk.shape[2] % 2 == 0

        downscaled_chunk = np.zeros([s // 2 for s in old_chunk.shape],
                                    dtype=np.float32)
        downscaled_chunk += old_chunk[::2, ::2, ::2]
        downscaled_chunk += old_chunk[1::2, ::2, ::2]
        downscaled_chunk += old_chunk[::2, 1::2, ::2]
        downscaled_chunk += old_chunk[1::2, 1::2, ::2]
        downscaled_chunk += old_chunk[::2, ::2, 1::2]
        downscaled_chunk += old_chunk[1::2, ::2, 1::2]
        downscaled_chunk += old_chunk[::2, 1::2, 1::2]
        downscaled_chunk += old_chunk[1::2, 1::2, 1::2]
        return np.asarray(np.round(downscaled_chunk / 8),
                          dtype=old_chunk.dtype)

    for sag_idx in range((new_size[0] - 1) // chunk_size[0] + 1):
        for cor_idx in range((new_size[1] - 1) // chunk_size[1] + 1):
            for ax_idx in range((new_size[2] - 1) // chunk_size[2] + 1):
                new_chunk = np.empty(
                    [min(chunk_size[0], new_size[0] - sag_idx * chunk_size[0]),
                     min(chunk_size[1], new_size[1] - cor_idx * chunk_size[1]),
                     min(chunk_size[2], new_size[2] - ax_idx * chunk_size[2])],
                    dtype=np.uint8  # TODO do not hard-code dtype
                    )
                new_chunk[:half_chunk[0], :half_chunk[1],
                          :half_chunk[2]] = (
                              load_and_downscale_old_chunk(sag_idx * 2,
                                                           cor_idx * 2,
                                                           ax_idx * 2))
                if new_chunk.shape[0] > half_chunk[0]:
                    new_chunk[half_chunk[0]:, :half_chunk[1],
                              :half_chunk[2]] = (
                                  load_and_downscale_old_chunk(sag_idx * 2 + 1,
                                                               cor_idx * 2,
                                                               ax_idx * 2))
                if new_chunk.shape[1] > half_chunk[1]:
                    new_chunk[:half_chunk[0], half_chunk[1]:,
                              :half_chunk[2]] = (
                                  load_and_downscale_old_chunk(sag_idx * 2,
                                                               cor_idx * 2 + 1,
                                                               ax_idx * 2))
                if (new_chunk.shape[0] > half_chunk[0]
                    and new_chunk.shape[1] > half_chunk[1]):
                    new_chunk[half_chunk[0]:, half_chunk[1]:,
                              :half_chunk[2]] = (
                                  load_and_downscale_old_chunk(sag_idx * 2 + 1,
                                                               cor_idx * 2 + 1,
                                                               ax_idx * 2))
                if new_chunk.shape[2] > half_chunk[2]:
                    new_chunk[:half_chunk[0], :half_chunk[1],
                              half_chunk[2]:] = (
                                  load_and_downscale_old_chunk(sag_idx * 2,
                                                               cor_idx * 2,
                                                               ax_idx * 2 + 1))
                if (new_chunk.shape[0] > half_chunk[0]
                    and new_chunk.shape[2] > half_chunk[2]):
                    new_chunk[half_chunk[0]:, :half_chunk[1],
                              half_chunk[2]:] = (
                                  load_and_downscale_old_chunk(sag_idx * 2 + 1,
                                                               cor_idx * 2,
                                                               ax_idx * 2 + 1))
                if (new_chunk.shape[1] > half_chunk[1]
                    and new_chunk.shape[2] > half_chunk[2]):
                    new_chunk[:half_chunk[0], half_chunk[1]:,
                              half_chunk[2]:] = (
                                  load_and_downscale_old_chunk(sag_idx * 2,
                                                               cor_idx * 2 + 1,
                                                               ax_idx * 2+ 1))
                if (new_chunk.shape[0] > half_chunk[0]
                    and new_chunk.shape[1] > half_chunk[1]
                    and new_chunk.shape[2] > half_chunk[2]):
                    new_chunk[half_chunk[0]:, half_chunk[1]:,
                              half_chunk[2]:] = (
                                  load_and_downscale_old_chunk(sag_idx * 2 + 1,
                                                               cor_idx * 2 + 1,
                                                               ax_idx * 2 + 1))

                new_chunk_name = npz_chunk_scheme.format(sag_idx, cor_idx,
                                                          ax_idx, key=new_key)
                print("Writing", new_chunk_name)
                os.makedirs(os.path.dirname(new_chunk_name), exist_ok=True)
                with open(new_chunk_name, "wb") as f:
                    np.savez(f, chunk=new_chunk)

    return new_scale_info


def make_jpeg_chunks(scale_info):
    import PIL.Image
    assert len(scale_info["chunk_sizes"]) == 1
    chunk_size = scale_info["chunk_sizes"][0]
    key = scale_info["key"]
    size = scale_info["size"]

    for sag_idx in range((size[0] - 1) // chunk_size[0] + 1):
        for cor_idx in range((size[1] - 1) // chunk_size[1] + 1):
            for ax_idx in range((size[2] - 1) // chunk_size[2] + 1):
                npz_chunk_filename = npz_chunk_scheme.format(sag_idx, cor_idx,
                                                             ax_idx, key=key)
                chunk = np.load(npz_chunk_filename)["chunk"]
                flattened_chunk = np.reshape(
                    np.moveaxis(chunk, [0, 1, 2], [2, 0, 1]),
                    (chunk.shape[2] * chunk.shape[1], chunk.shape[0]),
                    order='F')
                jpeg_chunk_filename = jpeg_chunk_scheme.format(
                    sag_idx * chunk_size[0],
                    sag_idx * chunk_size[0] + chunk.shape[0],
                    cor_idx * chunk_size[1],
                    cor_idx * chunk_size[1] + chunk.shape[1],
                    ax_idx * chunk_size[2],
                    ax_idx * chunk_size[2] + chunk.shape[2],
                    key=key)
                img = PIL.Image.fromarray(flattened_chunk)
                print("Writing", jpeg_chunk_filename)
                os.makedirs(os.path.dirname(jpeg_chunk_filename), exist_ok=True)
                img.save(jpeg_chunk_filename,
                         format="jpeg",
                         quality=95, optimize=True, progressive=True)


create_info_json()
#info = read_info_json()
# create_fullres_chunks(full_scale_info=info["scales"][0])
# for i in range(len(info["scales"]) - 1):
#     assert create_next_scale(info["scales"][i]) == info["scales"][i + 1]
# for i in reversed(range(len(info["scales"]))):
#     make_jpeg_chunks(info["scales"][i])
