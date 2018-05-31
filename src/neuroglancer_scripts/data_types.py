# Copyright (c) 2018 Forschungszentrum Juelich GmbH
# Author: Yann Leprince <y.leprince@fz-juelich.de>
#
# This software is made available under the MIT licence, see LICENCE.txt.

import logging

import numpy as np


logger = logging.getLogger(__name__)


NG_INTEGER_DATA_TYPES = ("uint8", "uint16", "uint32", "uint64")
NG_DATA_TYPES = NG_INTEGER_DATA_TYPES + ("float32",)


# TODO re-factor into a class:
# - implement reporting of non-preserved values (clipped / rounded)
# - implement optional scaling
# - print a warning for NaNs during float->int conversion
def get_chunk_dtype_transformer(input_dtype, output_dtype):
    input_dtype = np.dtype(input_dtype)
    output_dtype = np.dtype(output_dtype)
    if np.issubdtype(output_dtype, np.integer):
        output_min = np.iinfo(output_dtype).min
        output_max = np.iinfo(output_dtype).max
    else:
        output_min = 0.0
        output_max = 1.0

    work_dtype = np.promote_types(input_dtype, output_dtype)

    round_to_nearest = (
        np.issubdtype(output_dtype, np.integer)
        and not np.issubdtype(input_dtype, np.integer)
    )
    clip_values = (
        np.issubdtype(output_dtype, np.integer)
        and not np.can_cast(input_dtype, output_dtype, casting="safe")
    )

    logger.debug("dtype converter from %s to %s: "
                 "work_dtype=%s, round_to_nearest=%s, clip_values=%s",
                 input_dtype, output_dtype,
                 work_dtype, round_to_nearest, clip_values)
    if round_to_nearest:
        logger.warning("Values will be rounded to the nearest integer")
    if clip_values:
        logger.warning("Values will be clipped to the range [%s, %s]",
                       output_min, output_max)

    def chunk_transformer(chunk, preserve_input=True):
        assert np.can_cast(chunk.dtype, input_dtype, casting="equiv")
        if round_to_nearest or clip_values:
            chunk = np.array(chunk, dtype=work_dtype, copy=preserve_input)
            if round_to_nearest:
                np.rint(chunk, out=chunk)
            if clip_values:
                np.clip(chunk, output_min, output_max, out=chunk)
        return chunk.astype(output_dtype, casting="unsafe")

    return chunk_transformer
