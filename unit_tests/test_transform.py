# Copyright (c) 2018 CEA
# Author: Yann Leprince <yann.leprince@cea.fr>
#
# This software is made available under the MIT licence, see LICENCE.txt.

import numpy as np

from neuroglancer_scripts.transform import (
    nifti_to_neuroglancer_transform,
    matrix_as_compact_urlsafe_json,
)


def test_matrix_as_compact_urlsafe_json():
    mat = np.array([[1, 1.5], [2, 3], [0, -1]])
    assert matrix_as_compact_urlsafe_json(mat) == "[[1_1.5]_[2_3]_[0_-1]]"


def test_nifti_to_neuroglancer_transform():
    nifti_transform = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    voxel_size = (1.0, 1.0, 1.0)
    ng_transform = nifti_to_neuroglancer_transform(nifti_transform, voxel_size)
    assert np.array_equal(ng_transform, np.array([
        [1, 0, 0, -0.5],
        [0, 1, 0, -0.5],
        [0, 0, 1, -0.5],
        [0, 0, 0, 1]
    ]))
