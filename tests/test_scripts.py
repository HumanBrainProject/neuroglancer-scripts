# Copyright (c) 2018 Forschungszentrum Juelich GmbH
# Author: Yann Leprince <y.leprince@fz-juelich.de>
#
# This software is made available under the MIT licence, see LICENCE.txt.

import os
import subprocess

import pytest

# Check that the scripts import correctly
from neuroglancer_scripts.scripts import compute_scales
from neuroglancer_scripts.scripts import convert_chunks
from neuroglancer_scripts.scripts import generate_scales_info
from neuroglancer_scripts.scripts import mesh_to_precomputed
from neuroglancer_scripts.scripts import scale_stats
from neuroglancer_scripts.scripts import slices_to_precomputed
from neuroglancer_scripts.scripts import volume_to_precomputed

# TODO skip tests if git-lfs files are missing
@pytest.fixture(scope="module")
def examples_dir(request):
    return request.fspath / ".." / ".." / "examples"

def test_jubrain_example_greyscale(examples_dir, monkeypatch, tmpdir):
    jubrain_dir = examples_dir / "JuBrain"
    monkeypatch.chdir(tmpdir)
    assert subprocess.call([
        "volume-to-precomputed",
        "--generate-info",
        str(jubrain_dir / "colin27T1_seg.nii.gz"),
        str(tmpdir / "colin27T1_seg")
    ]) == 4  # 4 means manual intervention required (data_type)
    assert subprocess.call([
        "generate-scales-info",
        str(tmpdir / "colin27T1_seg" / "info_fullres.json"),
        str(tmpdir / "colin27T1_seg")
    ]) == 0
    assert subprocess.call([
        "volume-to-precomputed",
        str(jubrain_dir / "colin27T1_seg.nii.gz"),
        str(tmpdir / "colin27T1_seg")
    ]) == 0
    assert subprocess.call([
        "compute-scales",
        str(tmpdir / "colin27T1_seg")
    ]) == 0

def test_jubrain_example_MPM(examples_dir, monkeypatch, tmpdir):
    jubrain_dir = examples_dir / "JuBrain"
    monkeypatch.chdir(tmpdir)
    assert subprocess.call([
        "volume-to-precomputed",
        "--generate-info",
        str(jubrain_dir / "MPM.nii.gz"),
        str(tmpdir / "MPM")
    ]) == 0
    assert subprocess.call([
        "generate-scales-info",
        "--type=segmentation",
        "--encoding=compressed_segmentation",
        str(tmpdir / "MPM" / "info_fullres.json"),
        str(tmpdir / "MPM")
    ]) == 0
    assert subprocess.call([
        "volume-to-precomputed",
        str(jubrain_dir / "colin27T1_seg.nii.gz"),
        str(tmpdir / "MPM")
    ]) == 0
    assert subprocess.call([
        "compute-scales",
        "--downscaling-method=stride",  # for test speed
        str(tmpdir / "MPM")
    ]) == 0
