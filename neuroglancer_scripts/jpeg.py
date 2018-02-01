#! /usr/bin/env python3
#
# Copyright (c) 2016, 2017, 2018 Forschungszentrum Juelich GmbH
# Author: Yann Leprince <y.leprince@fz-juelich.de>
#
# This software is made available under the MIT licence, see LICENCE.txt.

import io

import numpy as np
import PIL.Image


def encode_chunk(chunk, jpeg_quality, jpeg_plane):
    assert 0 <= jpeg_quality <= 100
    assert jpeg_plane in ("xy", "xz")
    num_channels = chunk.shape[0]
    if jpeg_plane == "xy":
        reshaped_chunk = chunk.reshape(
            num_channels, chunk.shape[1] * chunk.shape[2], chunk.shape[3])
    elif jpeg_plane == "xz":
        reshaped_chunk = chunk.reshape(
            num_channels, chunk.shape[1], chunk.shape[2] * chunk.shape[3])

    if num_channels == 1:
        reshaped_chunk = np.squeeze(reshaped_chunk, 0)
    else:
        # Channels (RGB) need to be along the last axis for PIL
        reshaped_chunk = np.swapaxes(reshaped_chunk, 0, 3)

    img = PIL.Image.fromarray(reshaped_chunk)
    io_buf = io.BytesIO()
    img.save(io_buf, format="jpeg", quality=jpeg_quality,
             optimize=True, progressive=True)
    return io_buf.getvalue()
