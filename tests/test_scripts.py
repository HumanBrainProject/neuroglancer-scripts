# Copyright (c) 2018 Forschungszentrum Juelich GmbH
# Author: Yann Leprince <y.leprince@fz-juelich.de>
#
# This software is made available under the MIT licence, see LICENCE.txt.

import gzip
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


@pytest.fixture(scope="module")
def examples_dir(request):
    return request.fspath / ".." / ".." / "examples"


def test_jubrain_example_MPM(examples_dir, tmpdir):
    input_nifti = examples_dir / "JuBrain" / "MPM.nii.gz"
    # The file may be present but be a git-lfs pointer file, so we need to open
    # it to make sure that it is the actual correct file.
    try:
        gzip.open(str(input_nifti)).read(348)
    except OSError as exc:
        pytest.skip("Cannot find a valid example file {0} for testing: {1}"
                    .format(input_nifti, exc))

    output_dir = tmpdir / "MPM"
    assert subprocess.call([
        "volume-to-precomputed",
        "--generate-info",
        str(input_nifti),
        str(output_dir)
    ]) == 0
    assert subprocess.call([
        "generate-scales-info",
        "--type=segmentation",
        "--encoding=compressed_segmentation",
        str(output_dir / "info_fullres.json"),
        str(output_dir)
    ]) == 0
    assert subprocess.call([
        "volume-to-precomputed",
        str(input_nifti),
        str(output_dir)
    ]) == 0
    assert subprocess.call([
        "compute-scales",
        "--downscaling-method=stride",  # for test speed
        str(output_dir)
    ]) == 0
    assert subprocess.call([
        "scale-stats",
        str(output_dir),
    ]) == 0
    assert subprocess.call([
        "convert-chunks",
        "--copy-info",
        str(output_dir),
        str(output_dir / "copy")
    ]) == 0
