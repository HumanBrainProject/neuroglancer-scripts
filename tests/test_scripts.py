# Copyright (c) 2018 Forschungszentrum Juelich GmbH
# Author: Yann Leprince <y.leprince@fz-juelich.de>
#
# This software is made available under the MIT licence, see LICENCE.txt.

import os

import pytest

from neuroglancer_scripts.scripts import compute_scales
from neuroglancer_scripts.scripts import convert_chunks
from neuroglancer_scripts.scripts import generate_scales_info
from neuroglancer_scripts.scripts import mesh_to_precomputed
from neuroglancer_scripts.scripts import scale_stats
from neuroglancer_scripts.scripts import slices_to_precomputed
from neuroglancer_scripts.scripts import volume_to_precomputed

@pytest.fixture(scope="module")
def examples_dir(request):
    return request.fspath / ".." / ".." / "examples"

def test_volume_conversion(examples_dir, monkeypatch, tmpdir):
    jubrain_dir = examples_dir / "JuBrain"
    monkeypatch.chdir(tmpdir)
    assert volume_to_precomputed.main([
        "volume-to-precomputed",
        "--generate-info",
        str(jubrain_dir / "colin27T1_seg.nii.gz"),
        "colin27T1_seg/"
    ]) in (0, 4, None)  # retcode=4 means manual intervention
    assert generate_scales_info.main([
        "generate-scales-info",
        "colin27T1_seg/info_fullres.json",
        "colin27T1_seg/"
    ]) in (0, None)
    assert volume_to_precomputed.main([
        "volume-to-precomputed",
        str(jubrain_dir / "colin27T1_seg.nii.gz"),
        "colin27T1_seg/"
    ]) in (0, None)
    assert compute_scales.main([
        "compute-scales",
        "colin27T1_seg/"
    ]) in (0, None)
