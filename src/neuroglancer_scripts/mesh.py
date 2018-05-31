# Copyright (c) 2018 CEA
# Copyright (c) 2018 Forschungszentrum Juelich GmbH
# Author: Yann Leprince <y.leprince@fz-juelich.de>
#
# This software is made available under the MIT licence, see LICENCE.txt.

"""I/O for meshes in formats understood by Neuroglancer.

Neuroglancer understands two file formats for meshes:

- A binary “precomputed” format for meshes that correspond to a
  ``segmentation`` layer.

- A sub-set of the legacy VTK ASCII format can be used with the ``vtk://``
  datasource to represent a mesh that is not associated with a voxel
  segmentation (``SingleMesh`` layer). Vertices may have arbitrary scalar
  attributes.
"""

import logging
import re
import struct

import numpy as np

import neuroglancer_scripts


__all__ = [
    "InvalidMeshDataError",
    "save_mesh_as_neuroglancer_vtk",
    "save_mesh_as_precomputed",
    "read_precomputed_mesh",
]


logger = logging.getLogger(__name__)


class InvalidMeshDataError(Exception):
    """Raised when mesh data cannot be decoded properly."""
    pass


def save_mesh_as_neuroglancer_vtk(file, vertices, triangles,
                                  vertex_attributes=None, title=""):
    """Store a mesh in VTK format such that it can be read by Neuroglancer.

    :param file: a file-like object opened in text mode (its ``write`` method
        will be called with :class:`str` objects).
    :param numpy.ndarray vertices: the list of vertices of the mesh. Must be
        convertible to an array of size Nx3, type ``float32``. Coordinates
        will be interpreted by Neuroglancer in nanometres.
    :param numpy.ndarray triangles: the list of triangles of the mesh. Must be
        convertible to an array of size Mx3, with integer data type.
    :param list vertex_attributes: an iterable containing a description of
        vertex attributes (see below).
    :param str title: a title (comment) for the dataset. Cannot contain \\n,
        will be truncated to 255 characters.
    :raises AssertionError: if the inputs do not match the constraints above

    Each element of ``vertex_attributes`` must be a mapping (e.g.
    :class:`dict`) with the following keys:
    name
       The name of the vertex attribute, as a :class:`str`. Cannot contain
       white space.

    values
        The values of the attribute. Must be convertible to an array of size N
        or NxC, where N is the number of vertices, and C is the number of
        channels for the attribute (between 1 and 4).

    The output uses a sub-set of the legacy VTK ASCII format, which can be read
    by Neuroglancer (as of
    https://github.com/google/neuroglancer/blob/a8ce681660864ab3ac7c1086c0b4262e40f24707/src/neuroglancer/datasource/vtk/parse.ts).
    """
    vertices = np.asarray(vertices)
    assert vertices.ndim == 2
    triangles = np.asarray(triangles)
    assert triangles.ndim == 2
    assert triangles.shape[1] == 3
    assert "\n" not in title
    file.write("# vtk DataFile Version 3.0\n")
    if title:
        title += ". "
    title += "Written by neuroglancer-scripts-{0}.".format(
        neuroglancer_scripts.__version__
    )
    file.write("{0}\n".format(title[:255]))
    file.write("ASCII\n")
    file.write("DATASET POLYDATA\n")
    file.write("POINTS {0:d} {1}\n".format(vertices.shape[0], "float"))
    if not np.can_cast(vertices.dtype, np.float32):
        # As of a8ce681660864ab3ac7c1086c0b4262e40f24707 Neuroglancer reads
        # everything as float32 anyway
        logger.warn("Vertex coordinates will be converted to float32")
    np.savetxt(file, vertices.astype(np.float32), fmt="%.9g")
    file.write("POLYGONS {0:d} {1:d}\n"
               .format(triangles.shape[0], 4 * triangles.shape[0]))
    np.savetxt(file, np.insert(triangles, 0, 3, axis=1), fmt="%d")
    if vertex_attributes:
        file.write("POINT_DATA {0:d}\n".format(vertices.shape[0]))
        for vertex_attribute in vertex_attributes:
            name = vertex_attribute["name"]
            assert re.match("\\s", name) is None
            values = np.asarray(vertex_attribute["values"])
            assert values.shape[0] == vertices.shape[0]
            if values.ndim == 1:
                values = values[:, np.newaxis]
            assert values.ndim == 2
            num_components = values.shape[1]
            if num_components > 4:
                logger.warn("The file will not be strictly valid VTK because "
                            "a SCALAR vertex attribute has more than 4 "
                            "components")
            if not np.can_cast(values.dtype, np.float32):
                # As of a8ce681660864ab3ac7c1086c0b4262e40f24707 Neuroglancer
                # reads everything as float32 anyway
                logger.warn("Data for the '{0}' vertex attribute will be "
                            "converted to float32".format(name))
            file.write("SCALARS {0} {1}".format(name, "float"))
            if num_components != 1:
                file.write(" {0:d}".format(num_components))
            file.write("\nLOOKUP_TABLE {0}\n".format("default"))
            np.savetxt(file, values.astype(np.float32), fmt="%.9g")


def save_mesh_as_precomputed(file, vertices, triangles):
    """Store a mesh in Neuroglancer pre-computed format.

    :param file: a file-like object opened in binary mode (its ``write`` method
        will be called with :class:`bytes` objects).
    :param numpy.ndarray vertices: the list of vertices of the mesh. Must be
        convertible to an array of size Nx3 and type ``float32``. Coordinates
        will be interpreted by Neuroglancer in nanometres.
    :param numpy.ndarray triangles: the list of triangles of the mesh. Must be
        convertible to an array of size Mx3 and ``uint32`` data type.
    :raises AssertionError: if the inputs do not match the constraints above
    """
    vertices = np.asarray(vertices)
    assert vertices.ndim == 2
    triangles = np.asarray(triangles)
    assert triangles.ndim == 2
    assert triangles.shape[1] == 3
    if not np.can_cast(vertices.dtype, "<f"):
        logger.warn("Vertex coordinates will be converted to float32")
    file.write(struct.pack("<I", vertices.shape[0]))
    file.write(vertices.astype("<f").tobytes(order="C"))
    file.write(triangles.astype("<I", casting="safe").tobytes(order="C"))


def read_precomputed_mesh(file):
    """Load a mesh in Neuroglancer pre-computed format.

    :param file: a file-like object opened in binary mode (its ``read`` method
        is expected to return :class:`bytes` objects).
    :returns tuple: a 2-tuple ``(vertices, triangles)``, where ``vertices`` is
        an array of size Nx3 and type ``float32`` containing the vertex
        coordinates expressed in nanometres; and ``triangles`` is  an array
        of size Mx3 and ``uint32`` data type.
    """
    num_vertices = struct.unpack("<I", file.read(4))[0]
    # TODO handle format errors
    #
    # Use frombuffer instead of numpy.fromfile, because the latter expects a
    # real file and performs direct I/O on file.fileno(), which can fail or
    # read garbage e.g. if the file is an instance of gzip.GzipFile.
    buf = file.read(4 * 3 * num_vertices)
    if len(buf) != 4 * 3 * num_vertices:
        raise InvalidMeshDataError("The precomputed mesh data is too short")
    vertices = np.reshape(
        np.frombuffer(buf, "<f"),
        (num_vertices, 3),
        order="C"
    )
    # BUG: this could easily exhaust memory if reading a large file that is not
    # in precomputed format.
    buf = file.read()
    if len(buf) % (3 * 4) != 0:
        raise InvalidMeshDataError("The size of the precomputed mesh data is "
                                   "not adequate")
    flat_triangles = np.frombuffer(buf, "<I")
    triangles = np.reshape(flat_triangles, (-1, 3), order="C")
    if np.any(triangles > num_vertices):
        raise InvalidMeshDataError("The mesh references nonexistent vertices")
    return (vertices, triangles)
