# Copyright (c) 2018 Forschungszentrum Juelich GmbH
# Author: Yann Leprince <y.leprince@fz-juelich.de>
#
# This software is made available under the MIT licence, see LICENCE.txt.

import numpy as np
import pytest

from neuroglancer_scripts.downscaling import (
    get_downscaler,
    add_argparse_options,
    Downscaler,
    StridingDownscaler,
    AveragingDownscaler,
    MajorityDownscaler
)


@pytest.mark.parametrize("method", ["average", "majority", "stride"])
@pytest.mark.parametrize("options", [
    {},
    {"outside_value": 1.0, "unknown_option": None},
])
def test_get_downscaler(method, options):
    d = get_downscaler(method, options)
    assert isinstance(d, Downscaler)


def test_get_downscaler_auto_image():
    d = get_downscaler("auto", info={"type": "image"})
    assert isinstance(d, AveragingDownscaler)


def test_get_downscaler_auto_segmentation():
    d = get_downscaler("auto", info={"type": "segmentation"})
    assert isinstance(d, StridingDownscaler)


def test_add_argparse_options():
    import argparse
    parser = argparse.ArgumentParser()
    add_argparse_options(parser)
    # Test default values
    args = parser.parse_args([])
    get_downscaler(args.downscaling_method, {"type": "image"}, vars(args))
    args = parser.parse_args(["--downscaling-method", "auto"])
    assert args.downscaling_method == "auto"
    # Test correct parsing
    args = parser.parse_args(["--downscaling-method", "average"])
    assert args.downscaling_method == "average"
    assert args.outside_value is None
    args = parser.parse_args(["--downscaling-method", "average",
                              "--outside-value", "255"])
    assert args.downscaling_method == "average"
    assert args.outside_value == 255.
    args = parser.parse_args(["--downscaling-method", "majority"])
    assert args.downscaling_method == "majority"
    args = parser.parse_args(["--downscaling-method", "stride"])
    assert args.downscaling_method == "stride"


@pytest.mark.parametrize("dx", [1, 2])
@pytest.mark.parametrize("dy", [1, 2])
@pytest.mark.parametrize("dz", [1, 2])
def test_dyadic_downscaling(dx, dy, dz):
    scaling_factors = dx, dy, dz
    lowres_chunk = np.arange(3 * 7 * 4 * 2, dtype="f").reshape(2, 4, 7, 3)
    upscaled_chunk = np.empty((2, dz * 4, dy * 7, dx * 3),
                              dtype=lowres_chunk.dtype)
    for x, y, z in np.ndindex(scaling_factors):
        upscaled_chunk[:, z::dz, y::dy, x::dx] = lowres_chunk

    # Shorten the chunk by 1 voxel in every direction where it was upscaled
    truncation_slicing = (np.s_[:],) + tuple(
        np.s_[:-1] if s == 2 else np.s_[:] for s in reversed(scaling_factors)
    )
    truncated_chunk = upscaled_chunk[truncation_slicing]

    d = StridingDownscaler()
    assert np.array_equal(d.downscale(upscaled_chunk, scaling_factors),
                          lowres_chunk)
    assert np.array_equal(d.downscale(truncated_chunk, scaling_factors),
                          lowres_chunk)

    d = AveragingDownscaler()
    assert np.array_equal(d.downscale(upscaled_chunk, scaling_factors),
                          lowres_chunk)
    assert np.array_equal(d.downscale(truncated_chunk, scaling_factors),
                          lowres_chunk)

    d = MajorityDownscaler()
    assert np.array_equal(d.downscale(upscaled_chunk, scaling_factors),
                          lowres_chunk)
    assert np.array_equal(d.downscale(truncated_chunk, scaling_factors),
                          lowres_chunk)


def test_averaging_downscaler_rounding():
    d = AveragingDownscaler()
    test_chunk = np.array([[1, 1], [1, 0]], dtype="uint8").reshape(1, 2, 2, 1)
    assert np.array_equal(d.downscale(test_chunk, (1, 2, 2)),
                          np.array([1], dtype="uint8").reshape(1, 1, 1, 1))
