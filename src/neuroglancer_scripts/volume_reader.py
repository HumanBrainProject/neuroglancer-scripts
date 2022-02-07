# Copyright (c) 2016–2018, Forschungszentrum Jülich GmbH
# Author: Yann Leprince <y.leprince@fz-juelich.de>
#
# This software is made available under the MIT licence, see LICENCE.txt.

import json
import logging

import numpy as np
import nibabel
import nibabel.orientations
from tqdm import tqdm

import neuroglancer_scripts.accessor
from neuroglancer_scripts.accessor import DataAccessError
import neuroglancer_scripts.data_types
from neuroglancer_scripts import precomputed_io
import neuroglancer_scripts.transform


__all__ = [
    "store_nibabel_image_to_fullres_info",
    "nibabel_image_to_info",
    "nibabel_image_to_precomputed",
    "volume_file_to_info",
    "volume_file_to_precomputed",
    "volume_to_precomputed",
]


logger = logging.getLogger(__name__)


# TODO turn return codes into exceptions
#
# TODO factor out redundant code with nibabel_image_to_precomputed
def store_nibabel_image_to_fullres_info(img,
                                        accessor,
                                        ignore_scaling=False,
                                        input_min=None,
                                        input_max=None,
                                        options={}):
    formatted_info, json_transform, input_dtype, imperfect_dtype = (
        nibabel_image_to_info(
            img,
            ignore_scaling=ignore_scaling,
            input_min=input_min,
            input_max=input_max,
            options=options
        ))
    try:
        accessor.store_file("info_fullres.json",
                            formatted_info.encode("utf-8"),
                            mime_type="application/json")
    except DataAccessError as exc:
        logger.critical("cannot write info_fullres.json: %s", exc)
        return 1
    logger.info("The metadata above was written to info_fullres.json. "
                "Please run generate-scales-info on that file "
                "to generate the 'info' file, then run this program "
                "again.")
    try:
        s = json.dumps(json_transform)
        accessor.store_file("transform.json", s.encode("utf-8"),
                            mime_type="application/json")
    except DataAccessError as exc:
        logger.error("cannot write transform.json: %s", exc)
    logger.info("Neuroglancer transform of the converted volume "
                "(written to transform.json):\n%s",
                neuroglancer_scripts.transform.matrix_as_compact_urlsafe_json(
                    json_transform))
    return 4 if imperfect_dtype else 0


def nibabel_image_to_info(img,
                          ignore_scaling=False,
                          input_min=None,
                          input_max=None,
                          options={}):
    shape = img.header.get_data_shape()

    proxy = img.dataobj
    if ignore_scaling:
        proxy._slope = 1.0
        proxy._inter = 0.0

    if input_max is not None:
        # In case scaling is used, usually the result will be provided by
        # nibabel as float64
        input_dtype = np.dtype(np.float64)
    else:
        # There is no guarantee that proxy.dtype exists, so we have to
        # read a value from the file to see the result of the scaling
        zero_index = tuple(0 for _ in shape)
        input_dtype = proxy[zero_index].dtype

    input_dtype, is_rgb = neuroglancer_scripts.data_types.get_dtype(
                            input_dtype)
    if is_rgb:
        shape = shape + (3,)

    logger.info("Input image shape is %s", shape)
    affine = img.affine
    voxel_sizes = nibabel.affines.voxel_sizes(affine)
    logger.info("Input voxel size is %s mm", voxel_sizes)

    logger.info("Detected input axis orientations %s+",
                "".join(nibabel.orientations.aff2axcodes(affine)))

    if input_dtype.name in neuroglancer_scripts.data_types.NG_DATA_TYPES:
        guessed_dtype = input_dtype.name
    else:
        guessed_dtype = "float32"
    formatted_info = """\
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
             data_type=guessed_dtype,
             size=list(shape[:3]),
             resolution=[vs * 1000000 for vs in voxel_sizes[:3]])

    info = json.loads(formatted_info)  # ensure well-formed JSON
    logger.info("the following info has been generated:\n%s", formatted_info)

    # We need to take the voxel scaling out of img.affine, and convert the
    # translation part from millimetres to nanometres.
    transform = np.empty((4, 4))
    transform[:, 0] = affine[:, 0] / voxel_sizes[0]
    transform[:, 1] = affine[:, 1] / voxel_sizes[1]
    transform[:, 2] = affine[:, 2] / voxel_sizes[2]
    transform[:3, 3] = affine[:3, 3] * 1000000
    transform[3, 3] = 1
    # Finally, compensate the half-voxel shift which is due to the
    # different conventions of Nifti and Neuroglancer.
    transform = neuroglancer_scripts.transform.nifti_to_neuroglancer_transform(
        transform, np.asarray(info["scales"][0]["resolution"]))
    json_transform = [list(row) for row in transform]

    imperfect_dtype = (input_dtype.name
                       not in neuroglancer_scripts.data_types.NG_DATA_TYPES)
    if imperfect_dtype:
        logger.warn("The %s data type is not supported by Neuroglancer. "
                    "float32 was set, please adjust if needed "
                    "(data_type must be one of %s). The values will be "
                    "rounded (if targeting an integer type) and cast "
                    "during the conversion.",
                    input_dtype.name,
                    neuroglancer_scripts.data_types.NG_DATA_TYPES)
    return formatted_info, json_transform, input_dtype, imperfect_dtype


def volume_to_precomputed(pyramid_writer, volume, chunk_transformer=None):
    info = pyramid_writer.info
    assert len(info["scales"][0]["chunk_sizes"]) == 1  # more not implemented
    chunk_size = info["scales"][0]["chunk_sizes"][0]  # in order x, y, z
    size = info["scales"][0]["size"]  # in order x, y, z
    dtype = np.dtype(info["data_type"]).newbyteorder("<")
    num_channels = info["num_channels"]

    # Volumes given by nibabel are using Fortran indexing (X, Y, Z, T)
    assert volume.shape[:3] == tuple(size)
    if len(volume.shape) > 3:
        assert volume.shape[3] == num_channels

    progress_bar = tqdm(
        total=(((size[0] - 1) // chunk_size[0] + 1)
               * ((size[1] - 1) // chunk_size[1] + 1)
               * ((size[2] - 1) // chunk_size[2] + 1)),
        desc="writing", unit="chunks", leave=True)
    for z_chunk_idx in range((size[2] - 1) // chunk_size[2] + 1):
        z_slicing = np.s_[
            chunk_size[2] * z_chunk_idx
            : min(chunk_size[2] * (z_chunk_idx + 1), size[2])
        ]
        for y_chunk_idx in range((size[1] - 1) // chunk_size[1] + 1):
            y_slicing = np.s_[
                chunk_size[1] * y_chunk_idx
                : min(chunk_size[1] * (y_chunk_idx + 1), size[1])
            ]
            for x_chunk_idx in range((size[0] - 1) // chunk_size[0] + 1):
                x_slicing = np.s_[
                    chunk_size[0] * x_chunk_idx
                    : min(chunk_size[0] * (x_chunk_idx + 1), size[0])
                ]
                if len(volume.shape) == 4:
                    chunk = volume[x_slicing, y_slicing, z_slicing, :]
                elif len(volume.shape) == 3:
                    chunk = volume[x_slicing, y_slicing, z_slicing]
                    chunk = chunk[..., np.newaxis]

                if chunk_transformer is not None:
                    chunk = chunk_transformer(chunk, preserve_input=False)

                chunk = np.moveaxis(chunk, (0, 1, 2, 3), (3, 2, 1, 0))
                assert chunk.size == ((x_slicing.stop - x_slicing.start)
                                      * (y_slicing.stop - y_slicing.start)
                                      * (z_slicing.stop - z_slicing.start)
                                      * num_channels)

                chunk_coords = (x_slicing.start, x_slicing.stop,
                                y_slicing.start, y_slicing.stop,
                                z_slicing.start, z_slicing.stop)
                pyramid_writer.write_chunk(
                    chunk.astype(dtype, casting="equiv"),
                    info["scales"][0]["key"], chunk_coords
                )
                progress_bar.update()


def nibabel_image_to_precomputed(img,
                                 precomputed_writer,
                                 ignore_scaling=False,
                                 input_min=None,
                                 input_max=None,
                                 load_full_volume=True,
                                 options={}):
    shape = img.header.get_data_shape()

    proxy = img.dataobj
    if ignore_scaling:
        proxy._slope = 1.0
        proxy._inter = 0.0

    if input_max is not None:
        # In case scaling is used, usually the result will be provided by
        # nibabel as float64
        input_dtype = np.dtype(np.float64)
    else:
        # There is no guarantee that proxy.dtype exists, so we have to
        # read a value from the file to see the result of the scaling
        zero_index = tuple(0 for _ in shape)
        input_dtype = proxy[zero_index].dtype

    affine = img.affine
    voxel_sizes = nibabel.affines.voxel_sizes(affine)

    info = precomputed_writer.info

    output_dtype = np.dtype(info["data_type"])
    info_voxel_sizes = 1e-6 * np.asarray(info["scales"][0]["resolution"])
    if not np.allclose(voxel_sizes, info_voxel_sizes):
        logger.warning("voxel size is inconsistent with resolution in the "
                       "info file (%s mm)",
                       " × ".join(str(sz) for sz in info_voxel_sizes))

    if not np.can_cast(input_dtype, output_dtype, casting="safe"):
        logger.warning("The volume has data type %s, but chunks will be "
                       "saved with %s. You should make sure that the cast "
                       "does not lose range/accuracy.",
                       input_dtype.name, output_dtype.name)

    # Scaling according to --input-min and --input-max. We modify the
    # slope/inter values used by Nibabel rather than re-implementing
    # post-scaling of the read data, in order to benefit from the clever
    # handling of data types by Nibabel
    if np.issubdtype(output_dtype, np.integer):
        output_min = np.iinfo(output_dtype).min
        output_max = np.iinfo(output_dtype).max
    else:
        output_min = 0.0
        output_max = 1.0
    if input_max is not None:
        if input_min is None:
            input_min = 0
        postscaling_slope = (output_max - output_min) / (input_max - input_min)
        postscaling_inter = output_min - input_min * postscaling_slope
        prescaling_slope = proxy.slope
        prescaling_inter = proxy.inter
        proxy._slope = prescaling_slope * postscaling_slope
        proxy._inter = prescaling_inter * postscaling_slope + postscaling_inter

    # Transformations applied to the voxel values
    chunk_transformer = (
        neuroglancer_scripts.data_types.get_chunk_dtype_transformer(
            input_dtype, output_dtype
        )
    )
    if load_full_volume:
        logger.info("Loading full volume to memory... ")
        volume = img.get_data()
    else:
        volume = proxy
    logger.info("Writing chunks... ")
    volume_to_precomputed(precomputed_writer, volume,
                          chunk_transformer=chunk_transformer)


def volume_file_to_precomputed(volume_filename,
                               dest_url,
                               ignore_scaling=False,
                               input_min=None,
                               input_max=None,
                               load_full_volume=True,
                               options={}):
    img = nibabel.load(volume_filename)
    dtype, is_rgb = neuroglancer_scripts.data_types.get_dtype_from_vol(
                        img.dataobj)
    if is_rgb:
        proxy = np.asarray(img.dataobj)
        new_proxy = proxy.view(dtype=np.uint8, type=np.ndarray)
        new_dataobj = np.stack([
            new_proxy[:, :, 0::3],
            new_proxy[:, :, 1::3],
            new_proxy[:, :, 2::3]
        ], axis=-1)
        img = nibabel.Nifti1Image(new_dataobj, img.affine)

    accessor = neuroglancer_scripts.accessor.get_accessor_for_url(
        dest_url, options
    )
    try:
        precomputed_writer = precomputed_io.get_IO_for_existing_dataset(
            accessor
        )
    except neuroglancer_scripts.accessor.DataAccessError as exc:
        logger.error("No 'info' file was found (%s). You can generate one by "
                     "running this program with the --generate-info option, "
                     "then using generate_scales_info.py on the result",
                     exc)
        return 1
    except ValueError as exc:  # TODO use specific exception for invalid JSON
        logger.error("Invalid 'info' file: %s", exc)
        return 1
    return nibabel_image_to_precomputed(img, precomputed_writer,
                                        ignore_scaling, input_min, input_max,
                                        load_full_volume, options)


def volume_file_to_info(volume_filename, dest_url,
                        ignore_scaling=False,
                        input_min=None,
                        input_max=None,
                        options={}):
    img = nibabel.load(volume_filename)
    accessor = neuroglancer_scripts.accessor.get_accessor_for_url(
        dest_url,
        accessor_options=options
    )
    return store_nibabel_image_to_fullres_info(
        img,
        accessor,
        ignore_scaling=ignore_scaling,
        input_min=input_min,
        input_max=input_max,
        options=options
    )
