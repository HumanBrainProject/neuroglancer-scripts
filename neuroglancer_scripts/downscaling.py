#! /usr/bin/env python3
#
# Copyright (c) 2016, 2017, 2018 Forschungszentrum Juelich GmbH
# Author: Yann Leprince <y.leprince@fz-juelich.de>
#
# This software is made available under the MIT licence, see LICENCE.txt.

import numpy as np


def get_downscaler(downscaling_method, options={}):
    if downscaling_method == "average":
        outside_value = options.get("outside_value")
        return AveragingDownscaler(outside_value)
    elif downscaling_method == "majority":
        return MajorityDownscaler()
    elif downscaling_method == "stride":
        return StridingDownscaler()


def add_argparse_options(parser):
    group = parser.add_argument_group("Options for downscaling")
    group.add_argument("--downscaling-method", default="average",
                       choices=("average", "majority", "stride"),
                       help='"average" is recommended for grey-level images, '
                       '"majority" for segmentation images. "stride" is the '
                       'fastest, but provides no protection against aliasing '
                       'artefacts.')
    group.add_argument("--outside-value", type=float, default=None,
                       help="padding value used by the 'average' downscaling "
                       "method for computing the voxels at the border. If "
                       "omitted, the volume is padded with its edge values.")


class StridingDownscaler:
    def check_factors(self, downscaling_factors):
        return True

    def downscale(self, chunk, downscaling_factors):
        return chunk[:,
                     ::downscaling_factors[2],
                     ::downscaling_factors[1],
                     ::downscaling_factors[0]
        ]


class AveragingDownscaler:
    def __init__(self, outside_value=None):
        if outside_value is None:
            self.padding_mode = "edge"
            self.pad_kwargs = {}
        else:
            self.padding_mode = "constant"
            self.pad_kwargs = {"constant_values": outside_value}

    def check_factors(self, downscaling_factors):
        return all(f in (1, 2) for f in downscaling_factors)

    def downscale(self, chunk, downscaling_factors):
        dtype = chunk.dtype
        chunk = chunk.astype(np.float32)  # unbounded type for arithmetic

        if downscaling_factors[2] == 2:
            if chunk.shape[1] % 2 != 0:
                chunk = np.pad(chunk, ((0, 0), (0, 1), (0, 0), (0, 0)),
                               self.padding_mode, **self.pad_kwargs)
            chunk = (chunk[:, ::2, :, :] + chunk[:, 1::2, :, :]) * 0.5

        if downscaling_factors[1] == 2:
            if chunk.shape[2] % 2 != 0:
                chunk = np.pad(chunk, ((0, 0), (0, 0), (0, 1), (0, 0)),
                               self.padding_mode, **self.pad_kwargs)
            chunk = (chunk[:, :, ::2, :] + chunk[:, :, 1::2, :]) * 0.5

        if downscaling_factors[0] == 2:
            if chunk.shape[3] % 2 != 0:
                chunk = np.pad(chunk, ((0, 0), (0, 0), (0, 0), (0, 1)),
                               self.padding_mode, **self.pad_kwargs)
            chunk = (chunk[:, :, :, ::2] + chunk[:, :, :, 1::2]) * 0.5

        return chunk.astype(dtype)


class MajorityDownscaler:
    def check_factors(self, downscaling_factors):
        return True

    def downscale(self, chunk, downscaling_factors):
        # This could be optimized a lot (clever iteration with nditer, Cython)
        new_chunk = np.empty(
            (chunk.shape[0],
             (chunk.shape[1] - 1) // downscaling_factors[2] + 1,
             (chunk.shape[2] - 1) // downscaling_factors[1] + 1,
             (chunk.shape[3] - 1) // downscaling_factors[0] + 1),
            dtype=chunk.dtype
        )
        for t, z, y, x in np.ndindex(*new_chunk.shape):
            zd = z * downscaling_factors[2]
            yd = y * downscaling_factors[1]
            xd = x * downscaling_factors[0]
            block = chunk[t,
                          zd:(zd + downscaling_factors[2]),
                          yd:(yd + downscaling_factors[1]),
                          xd:(xd + downscaling_factors[0])
            ]

            labels, counts = np.unique(block.flat, return_counts=True)
            new_chunk[t, z, y, x] = labels[np.argmax(counts)]

        return new_chunk
