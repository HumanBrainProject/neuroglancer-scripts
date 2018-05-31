# Copyright (c) 2018 CEA
# Author: Yann Leprince <yann.leprince@cea.fr>
#
# This software is made available under the MIT licence, see LICENCE.txt.

import logging

from neuroglancer_scripts.dyadic_pyramid import (
    choose_unit_for_key,
    fill_scales_for_dyadic_pyramid,
)


def test_choose_unit_for_key():
    assert choose_unit_for_key(1e-3) == "pm"
    assert choose_unit_for_key(1) == "nm"
    assert choose_unit_for_key(1e3) == "um"
    assert choose_unit_for_key(1e6) == "mm"
    assert choose_unit_for_key(1e9) == "m"


def test_fill_scales_for_dyadic_pyramid_small_volume():
    info = fill_scales_for_dyadic_pyramid({"scales": [{
        "size": [1, 1, 1],
        "resolution": [1, 1, 1],
    }]})
    assert len(info["scales"]) > 0
    assert info["scales"][0]["size"] == [1, 1, 1]
    assert info["scales"][0]["resolution"] == [1, 1, 1]
    assert "chunk_sizes" in info["scales"][0]


def test_fill_scales_for_dyadic_pyramid_simple_isotropic():
    info = fill_scales_for_dyadic_pyramid({"scales": [{
        "size": [256, 256, 256],
        "resolution": [1e6, 1e6, 1e6],
    }]}, target_chunk_size=64)
    assert info["scales"][0]["size"] == [256, 256, 256]
    assert info["scales"][0]["resolution"] == [1e6, 1e6, 1e6]
    assert info["scales"][0]["chunk_sizes"] == [[64, 64, 64]]
    assert info["scales"][1]["size"] == [128, 128, 128]
    assert info["scales"][1]["resolution"] == [2e6, 2e6, 2e6]
    assert info["scales"][1]["chunk_sizes"] == [[64, 64, 64]]


def test_fill_scales_for_dyadic_pyramid_max_scales():
    info = fill_scales_for_dyadic_pyramid({"scales": [{
        "size": [1024, 1024, 1024],
        "resolution": [1e6, 1e6, 1e6],
    }]}, target_chunk_size=64, max_scales=2)
    assert len(info["scales"]) == 2


def chunk_extent_anisotropy(resolution, chunk_size):
    chunk_extent = [r * cs for r, cs in zip(resolution, chunk_size)
                    if cs > 1]
    return max(chunk_extent) / min(chunk_extent)


def test_fill_scales_for_dyadic_pyramid_simple_anisotropic():
    info = fill_scales_for_dyadic_pyramid({"scales": [{
        "size": [512, 256, 256],
        "resolution": [1e6, 2e6, 2e6],
    }]}, target_chunk_size=64)
    assert info["scales"][0]["size"] == [512, 256, 256]
    assert info["scales"][0]["resolution"] == [1e6, 2e6, 2e6]
    assert chunk_extent_anisotropy(info["scales"][0]["resolution"],
                                   info["scales"][0]["chunk_sizes"][0]) < 2
    assert info["scales"][1]["size"] == [256, 256, 256]
    assert info["scales"][1]["resolution"] == [2e6, 2e6, 2e6]
    assert info["scales"][1]["chunk_sizes"] == [[64, 64, 64]]
    assert info["scales"][2]["size"] == [128, 128, 128]
    assert info["scales"][2]["resolution"] == [4e6, 4e6, 4e6]
    assert info["scales"][2]["chunk_sizes"] == [[64, 64, 64]]


def test_fill_scales_for_dyadic_pyramid_extremely_anisotropic():
    info = fill_scales_for_dyadic_pyramid({"scales": [{
        "size": [256000000, 256, 256],
        "resolution": [1, 1e6, 1e6],
    }]}, target_chunk_size=64)
    assert info["scales"][0]["size"] == [256000000, 256, 256]
    assert info["scales"][0]["resolution"] == [1, 1e6, 1e6]
    assert chunk_extent_anisotropy(info["scales"][0]["resolution"],
                                   info["scales"][0]["chunk_sizes"][0]) < 2
    assert info["scales"][0]["chunk_sizes"] == [[64 ** 3, 1, 1]]


def test_fill_scales_for_dyadic_pyramid_extra_scales(caplog):
    info = fill_scales_for_dyadic_pyramid({"scales": [
        {
            "size": [256, 256, 256],
            "resolution": [1e6, 1e6, 1e6],
        },
        {
            "size": [64, 64, 128],
            "resolution": [4e6, 4e6, 2e6],
        }]}, target_chunk_size=64)
    # ensure that a warning is printed
    assert any(r.levelno == logging.WARNING for r in caplog.records)
    assert info["scales"][0]["size"] == [256, 256, 256]
    assert info["scales"][0]["resolution"] == [1e6, 1e6, 1e6]
    assert info["scales"][0]["chunk_sizes"] == [[64, 64, 64]]
    # ensure that the second scale was ignored
    assert info["scales"][1]["size"] == [128, 128, 128]
    assert info["scales"][1]["resolution"] == [2e6, 2e6, 2e6]
    assert info["scales"][1]["chunk_sizes"] == [[64, 64, 64]]
