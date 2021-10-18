# Copyright (c) 2018 Forschungszentrum Juelich GmbH
# Author: Yann Leprince <y.leprince@fz-juelich.de>
#
# This software is made available under the MIT licence, see LICENCE.txt.

import logging

import numpy as np


__all__ = [
    "NG_DATA_TYPES",
    "NG_INTEGER_DATA_TYPES",
    "get_chunk_dtype_transformer",
]


logger = logging.getLogger(__name__)

NG_MULTICHANNEL_DATATYPES = (('R', 'G', 'B'),)
NG_INTEGER_DATA_TYPES = ("uint8", "uint16", "uint32", "uint64")
NG_DATA_TYPES = NG_INTEGER_DATA_TYPES + ("float32",)


# TODO re-factor into a class:
# - implement reporting of non-preserved values (clipped / rounded)
# - implement optional scaling
# - print a warning for NaNs during float->int conversion
def get_chunk_dtype_transformer(input_dtype, output_dtype, warn=True):
    """

    .. note::
        Conversion to uint64 may result in loss of precision, because of a
        known bug / approximation in NumPy, where dtype promotion between a
        64-bit (u)int and any float will return float64, even though this type
        can only hold all integers up to 2**53 (see e.g.
        https://github.com/numpy/numpy/issues/8851).
    """
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
    if warn and round_to_nearest:
        logger.warning("Values will be rounded to the nearest integer")
    if warn and clip_values:
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


def get_dtype_from_vol(volume):
    zero_index = tuple(0 for _ in volume.shape)
    return get_dtype(volume[zero_index].dtype)


def get_dtype(input_dtype):
    if input_dtype.names is None:
        return input_dtype, False
    if input_dtype.names not in NG_MULTICHANNEL_DATATYPES:
        err = 'tuple datatype {} not yet supported'.format(input_dtype.names)
        raise NotImplementedError(err)
    for index, value in enumerate(input_dtype.names):
        err = 'Multichanneled datatype should have the same datatype'
        assert input_dtype[index].name == input_dtype[0].name, err
    return input_dtype[0], True
