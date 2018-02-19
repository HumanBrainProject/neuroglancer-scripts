# Copyright (c) 2018 Forschungszentrum Juelich GmbH
# Author: Yann Leprince <y.leprince@fz-juelich.de>
#
# This software is made available under the MIT licence, see LICENCE.txt.

import json

import numpy as np


def nifti_to_neuroglancer_transform(nifti_transformation_matrix, voxel_size):
    """Compensate the half-voxel shift between Neuroglancer and Nifti.

    Nifti specifies that the transformation matrix (legacy, qform, or sform)
    gives the spatial coordinates of the *centre* of a voxel, while the
    Neuroglancer "transform" matrix specifies the *corner* of voxels.

    This function compensates the resulting half-voxel shift by adjusting the
    translation parameters accordingly.
    """
    ret = np.copy(nifti_transformation_matrix)
    ret[:3, 3] -= np.dot(ret[:3, :3], 0.5 * np.asarray(voxel_size))
    return ret


def matrix_as_compact_urlsafe_json(matrix):
    # Transform tre matrix, transforming numbers whose floating-point
    # representation has a training .0 to integers
    array = [[int(x) if str(x).endswith(".0") and int(x) == x
              else x for x in row] for row in matrix]
    return json.dumps(array, indent=None, separators=('_', ':'))
