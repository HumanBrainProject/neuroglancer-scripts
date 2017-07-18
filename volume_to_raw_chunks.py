#! /usr/bin/env python3
#
# Copyright (c) 2016, 2017, Forschungszentrum Juelich GmbH
# Author: Yann Leprince <y.leprince@fz-juelich.de>
#
# This software is made available under the MIT licence, see LICENCE.txt.

import gzip
import json
import logging
import os
import os.path
import sys

import numpy as np
import nibabel
import nibabel.orientations
from tqdm import tqdm


logging.basicConfig(format='%(message)s', level=logging.INFO)


NG_DATA_TYPES = ("uint8", "uint16", "uint32", "uint64", "float32")

RAW_CHUNK_PATTERN = "{key}/{0}-{1}/{2}-{3}/{4}-{5}"


def nifti_to_neuroglancer_transform(nifti_transformation_matrix, voxel_size):
    """Compensate the half-voxel shift introduced by Neuroglancer for Nifti data

    Nifti specifies that the transformation matrix (legacy, qform, or sform)
    gives the spatial coordinates of the *centre* of a voxel, while the
    Neuroglancer "transform" matrix specifies the *corner* of voxels.

    This function compensates the resulting half-voxel shift by adjusting the
    translation parameters accordingly.
    """
    ret = np.copy(nifti_transformation_matrix)
    ret[:3, 3] -= np.dot(ret[:3, :3], 0.5 * np.asarray(voxel_size))
    return ret


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

    progress_bar = tqdm(
        total=(((size[0] - 1) // chunk_size[0] + 1)
               * ((size[1] - 1) // chunk_size[1] + 1)
               * ((size[2] - 1) // chunk_size[2] + 1)),
        desc="writing", unit="chunks", leave=True)
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
                progress_bar.update()


def volume_file_to_raw_chunks(volume_filename,
                              generate_info=False,
                              ignore_scaling=False):
    """Convert from neuro-imaging formats to pre-computed raw chunks"""
    img = nibabel.load(volume_filename)
    shape = img.header.get_data_shape()
    logging.info("Input image shape is %s", shape)
    affine = img.affine
    voxel_sizes = nibabel.affines.voxel_sizes(affine)
    logging.info("Input voxel size is %s mm", voxel_sizes)

    logging.info("Detected input axis orientations %s+",
                 "".join(nibabel.orientations.aff2axcodes(affine)))

    try:
        with open("info") as f:
            info = json.load(f)
    except:
        generate_info=True

    if generate_info:
        header_info = """\
{{
    "type": "image",
    "num_channels": {num_channels},
    "data_type": "{data_type}",
    "scales": [
        {{
            "encoding": "raw",
            "size": {size},
            "resolution": {resolution},
            "voxel_offset": [0, 0, 0]
        }}
    ]
}}""".format(num_channels=shape[3] if len(shape) >= 4 else 1,
            data_type=img.header.get_data_dtype().name,
            size=list(shape[:3]),
            resolution=[vs * 1000000 for vs in voxel_sizes[:3]])
        logging.info("Please generate the info file with "
                     "generate_scales_info.py using the information below, "
                     "then run this program again")
        json.loads(header_info)  # ensure well-formed JSON
        print(header_info)
        if img.header.get_data_dtype().name not in NG_DATA_TYPES:
            logging.error("%s data type is not supported by Neuroglancer. "
                          "You must set data_type to one of %s. The data "
                          "will be cast during conversion.",
                          img.header.get_data_dtype().name, NG_DATA_TYPES)
            # return code indicating that manual intervention is needed
            return 4
        # return code indicating that ready-to-use info was printed
        return 3

    info_voxel_sizes = 0.000001 * np.asarray(info["scales"][0]["resolution"])
    if not np.allclose(voxel_sizes, info_voxel_sizes):
        logging.warning("voxel size is inconsistent with resolution in the "
                        "info file(%s nm)", info_voxel_sizes)

    if ignore_scaling:
        img.header.set_slope_inter(None)

    logging.info("Loading volume...")
    volume = img.get_data()

    if volume.dtype.name != info["data_type"]:
        logging.info("The volume has data type %s, but chunks will be saved "
                     "with %s. You should make sure that the cast does not "
                     "lose range/accuracy.",
                     volume.dtype.name, info["data_type"])

    logging.info("Writing chunks... ")
    volume_to_raw_chunks(info, volume)

    # This is the affine of the converted volume, print it at the end so it
    # does not get lost in scrolling.
    #
    # We need to take the voxel scaling out of img.affine, and convert the
    # translation part from millimetres to nanometres.
    transform = np.empty((4, 4))
    transform[:, 0] = affine[:, 0] / voxel_sizes[0]
    transform[:, 1] = affine[:, 1] / voxel_sizes[1]
    transform[:, 2] = affine[:, 2] / voxel_sizes[2]
    transform[:3, 3] = affine[:3, 3] * 1000000
    transform[3, 3] = 1
    # Finally, compensate the half-voxel shift which is due to the different
    # conventions of Nifti and Neuroglancer.
    transform = nifti_to_neuroglancer_transform(
        transform, np.asarray(info["scales"][0]["resolution"]))
    json_transform = [list(row) for row in transform]
    logging.info("Neuroglancer transform of the converted volume:\n%s",
                 json.dumps(json_transform))
                 #np.array2string(transform, separator=", "))


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
    parser.add_argument("--ignore-scaling", action="store_true")
    parser.add_argument("--generate-info", action="store_true")
    args = parser.parse_args(argv[1:])
    return args


def main(argv):
    """The script's entry point."""
    args = parse_command_line(argv)
    return volume_file_to_raw_chunks(args.volume_filename,
                                     generate_info=args.generate_info,
                                     ignore_scaling=args.ignore_scaling) or 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
