# Copyright (c) 2018 CEA
# Author: Yann Leprince <y.leprince@fz-juelich.de>
#
# This software is made available under the MIT licence, see LICENCE.txt.

import gzip
import io

import numpy as np

from neuroglancer_scripts.mesh import *


def dummy_mesh(num_vertices=4, num_triangles=3):
    vertices = np.reshape(
        np.arange(3 * num_vertices, dtype=np.float32),
        (num_vertices, 3)
    )
    triangles = np.reshape(
        np.arange(3 * num_triangles, dtype=np.uint32),
        (num_triangles, 3)
    ) % num_vertices
    return vertices, triangles


def test_precomputed_mesh_roundtrip():
    vertices, triangles = dummy_mesh()
    file = io.BytesIO()
    save_mesh_as_precomputed(file, vertices, triangles)
    file.seek(0)
    vertices2, triangles2 = read_precomputed_mesh(file)
    assert np.array_equal(vertices, vertices2)
    assert np.array_equal(triangles, triangles2)


def test_precomputed_mesh_gzip_file_roundtrip():
    vertices, triangles = dummy_mesh()
    bytes_io = io.BytesIO()
    with gzip.GzipFile(fileobj=bytes_io, mode="wb") as f:
        save_mesh_as_precomputed(f, vertices, triangles)
    buf = bytes_io.getvalue()
    with gzip.GzipFile(fileobj=io.BytesIO(buf), mode="rb") as f:
        vertices2, triangles2 = read_precomputed_mesh(f)
    assert np.array_equal(vertices, vertices2)
    assert np.array_equal(triangles, triangles2)
