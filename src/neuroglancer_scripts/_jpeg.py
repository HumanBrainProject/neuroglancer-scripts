#! /usr/bin/env python3
#
# Copyright (c) 2016, 2017, 2018 Forschungszentrum Juelich GmbH
# Author: Yann Leprince <y.leprince@fz-juelich.de>
#
# This software is made available under the MIT licence, see LICENCE.txt.

import io

import numpy as np
import PIL.Image


from neuroglancer_scripts.chunk_encoding import InvalidFormatError


def encode_chunk(chunk, jpeg_quality, jpeg_plane):
    assert 0 <= jpeg_quality <= 100
    assert jpeg_plane in ("xy", "xz")
    num_channels = chunk.shape[0]
    if jpeg_plane == "xy":
        reshaped_chunk = chunk.reshape(
            num_channels, chunk.shape[1] * chunk.shape[2], chunk.shape[3])
    else:  # jpeg_plane == "xz":
        reshaped_chunk = chunk.reshape(
            num_channels, chunk.shape[1], chunk.shape[2] * chunk.shape[3])

    if num_channels == 1:
        reshaped_chunk = np.squeeze(reshaped_chunk, 0)
    else:
        # Channels (RGB) need to be along the last axis for PIL
        reshaped_chunk = np.moveaxis(reshaped_chunk, 0, -1)

    img = PIL.Image.fromarray(reshaped_chunk)
    io_buf = io.BytesIO()
    # Chroma sub-sampling is disabled because it can create strong artefacts at
    # the border where the chunk size is odd. Progressive is enabled because it
    # generally creates smaller JPEG files.
    img.save(io_buf, format="jpeg", quality=jpeg_quality,
             optimize=True, progressive=True, subsampling=0)
    return io_buf.getvalue()


def decode_chunk(buf, chunk_size, num_channels):
    io_buf = io.BytesIO(buf)
    try:
        img = PIL.Image.open(io_buf)
    except Exception as exc:
        raise InvalidFormatError(
            "The JPEG-encoded chunk could not be decoded: {0}"
            .format(exc)) from exc

    if num_channels == 1 and img.mode != "L":
        raise InvalidFormatError(
            "The JPEG chunk is encoded with mode={0} instead of L"
            .format(img.mode))
    if num_channels == 3 and img.mode != "RGB":
        raise InvalidFormatError(
            "The JPEG chunk is encoded with mode={0} instead of RGB"
            .format(img.mode))

    flat_chunk = np.asarray(img)
    if num_channels == 3:
        # RGB channels are read by PIL along the last axis
        flat_chunk = np.moveaxis(flat_chunk, -1, 0)
    try:
        chunk = flat_chunk.reshape(num_channels,
                                   chunk_size[2], chunk_size[1], chunk_size[0])
    except Exception as exc:
        raise InvalidFormatError("The JPEG-encoded chunk has an incompatible "
                                 "shape ({0} elements, expecting {1})"
                                 .format(flat_chunk.size // num_channels,
                                         np.prod(chunk_size)))
    return chunk
