#! /usr/bin/env python3
#
# Copyright (c) 2016, 2017, 2018 Forschungszentrum Juelich GmbH
# Author: Yann Leprince <y.leprince@fz-juelich.de>
#
# This software is made available under the MIT licence, see LICENCE.txt.

"""Downscaling is used to create a multi-resolution image pyramid.

The central component here is the :class:`Downscaler` base class. Use
:func:`get_downscaler` for instantiating a concrete downscaler object.
"""

import numpy as np

from neuroglancer_scripts.data_types import get_chunk_dtype_transformer
from neuroglancer_scripts.utils import ceil_div

__all__ = [
    "get_downscaler",
    "add_argparse_options",
    "Downscaler",
    "StridingDownscaler",
    "AveragingDownscaler",
    "MajorityDownscaler",
]


def get_downscaler(downscaling_method, info=None, options={}):
    """Create a downscaler object.

    :param str downscaling_method: one of ``"average"``, ``"majority"``, or
                                   ``"stride"``
    :param dict options: options passed to the downscaler as kwargs.
    :returns: an instance of a sub-class of :class:`Downscaler`
    :rtype: Downscaler
    """
    if downscaling_method == "auto":
        if info["type"] == "image":
            return get_downscaler("average", info=None, options=options)
        else:  # info["type"] == "segmentation":
            return get_downscaler("stride", info=None, options=options)
    elif downscaling_method == "average":
        outside_value = options.get("outside_value")
        return AveragingDownscaler(outside_value)
    elif downscaling_method == "majority":
        return MajorityDownscaler()
    elif downscaling_method == "stride":
        return StridingDownscaler()
    else:
        raise NotImplementedError("invalid downscaling method "
                                  + downscaling_method)


def add_argparse_options(parser):
    """Add command-line options for downscaling.

    :param argparse.ArgumentParser parser: an argument parser

    The downscaling options can be obtained from command-line arguments with
    :func:`add_argparse_options` and passed to :func:`get_downscaler`::

        import argparse
        parser = argparse.ArgumentParser()
        add_argparse_options(parser)
        args = parser.parse_args()
        get_downscaler(args.downscaling_method, vars(args))
    """
    group = parser.add_argument_group("Options for downscaling")
    group.add_argument("--downscaling-method", default="auto",
                       choices=("auto", "average", "majority", "stride"),
                       help='The default is "auto", which chooses '
                       '"average" or "stride" depending on the "type" '
                       'attribute of the dataset (for "image" or '
                       '"segmentation", respectively). "average" is '
                       'recommended for grey-level images. "majority" is a '
                       'high-quality, but very slow method for segmentation '
                       'images. "stride" is fastest, but provides no '
                       'protection against aliasing artefacts.')
    group.add_argument("--outside-value", type=float, default=None,
                       help='padding value used by the "average" downscaling '
                       "method for computing the voxels at the border. If "
                       "omitted, the volume is padded with its edge values.")


class Downscaler:
    """Base class for downscaling algorithms."""

    def check_factors(self, downscaling_factors):
        """Test support for given downscaling factors.

        Subclasses must override this method if they do not support any
        combination of integer downscaling factors.

        :param downscaling_factors: sequence of integer downscaling factors
                                    (Dx, Dy, Dz)
        :type downscaling_factors: :class:`tuple` of :class:`int`
        :returns: whether the provided downscaling factors are supported
        :rtype: bool
        """
        return (
            len(downscaling_factors) == 3
            and all(isinstance(f, int) and 1 <= f for f in downscaling_factors)
        )

    def downscale(self, chunk, downscaling_factors):
        """Downscale a chunk according to the provided factors.

        :param numpy.ndarray chunk: chunk with (C, Z, Y, X) indexing
        :param downscaling_factors: sequence of integer downscaling factors
                                    (Dx, Dy, Dz)
        :type downscaling_factors: tuple
        :returns: the downscaled chunk, with shape ``(C, ceil_div(Z, Dz),
                  ceil_div(Y, Dy), ceil_div(X, Dx))``
        :rtype: numpy.ndarray
        :raises NotImplementedError: if the downscaling factors are unsupported
        """
        raise NotImplementedError


class StridingDownscaler(Downscaler):
    """Downscale using striding.

    This is a fast, low-quality downscaler that provides no protection against
    aliasing artefacts. It supports arbitrary downscaling factors.
    """
    def downscale(self, chunk, downscaling_factors):
        if not self.check_factors(downscaling_factors):
            raise NotImplementedError
        return chunk[:,
                     ::downscaling_factors[2],
                     ::downscaling_factors[1],
                     ::downscaling_factors[0]]


class AveragingDownscaler(Downscaler):
    """Downscale by a factor of two in any given direction, with averaging.

    This downscaler is suitable for grey-level images.

    .. todo::
       Use code from the neuroglancer module to support arbitrary factors.
    """
    def __init__(self, outside_value=None):
        if outside_value is None:
            self.padding_mode = "edge"
            self.pad_kwargs = {}
        else:
            self.padding_mode = "constant"
            self.pad_kwargs = {"constant_values": outside_value}

    def check_factors(self, downscaling_factors):
        return (
            len(downscaling_factors) == 3
            and all(f in (1, 2) for f in downscaling_factors)
        )

    def downscale(self, chunk, downscaling_factors):
        if not self.check_factors(downscaling_factors):
            raise NotImplementedError
        dtype = chunk.dtype
        # Use a floating-point type for arithmetic
        work_dtype = np.promote_types(chunk.dtype, np.float64)
        chunk = chunk.astype(work_dtype, casting="safe")

        half = work_dtype.type(0.5)

        if downscaling_factors[2] == 2:
            if chunk.shape[1] % 2 != 0:
                chunk = np.pad(chunk, ((0, 0), (0, 1), (0, 0), (0, 0)),
                               self.padding_mode, **self.pad_kwargs)
            chunk = half * (chunk[:, ::2, :, :] + chunk[:, 1::2, :, :])

        if downscaling_factors[1] == 2:
            if chunk.shape[2] % 2 != 0:
                chunk = np.pad(chunk, ((0, 0), (0, 0), (0, 1), (0, 0)),
                               self.padding_mode, **self.pad_kwargs)
            chunk = half * (chunk[:, :, ::2, :] + chunk[:, :, 1::2, :])

        if downscaling_factors[0] == 2:
            if chunk.shape[3] % 2 != 0:
                chunk = np.pad(chunk, ((0, 0), (0, 0), (0, 0), (0, 1)),
                               self.padding_mode, **self.pad_kwargs)
            chunk = half * (chunk[:, :, :, ::2] + chunk[:, :, :, 1::2])

        dtype_converter = get_chunk_dtype_transformer(work_dtype, dtype,
                                                      warn=False)
        return dtype_converter(chunk)


class MajorityDownscaler(Downscaler):
    """Downscaler using majority voting.

    This downscaler is suitable for label images.

    .. todo::
       The majority downscaler could be *really* optimized (clever iteration
       with nditer, Cython, countless for appropriate cases)
    """
    def downscale(self, chunk, downscaling_factors):
        if not self.check_factors(downscaling_factors):
            raise NotImplementedError
        new_chunk = np.empty(
            (chunk.shape[0],
             ceil_div(chunk.shape[1], downscaling_factors[2]),
             ceil_div(chunk.shape[2], downscaling_factors[1]),
             ceil_div(chunk.shape[3], downscaling_factors[0])),
            dtype=chunk.dtype
        )
        for t, z, y, x in np.ndindex(*new_chunk.shape):
            zd = z * downscaling_factors[2]
            yd = y * downscaling_factors[1]
            xd = x * downscaling_factors[0]
            block = chunk[t,
                          zd:(zd + downscaling_factors[2]),
                          yd:(yd + downscaling_factors[1]),
                          xd:(xd + downscaling_factors[0])]

            labels, counts = np.unique(block.flat, return_counts=True)
            new_chunk[t, z, y, x] = labels[np.argmax(counts)]

        return new_chunk
