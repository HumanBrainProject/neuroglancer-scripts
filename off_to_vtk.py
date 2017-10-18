#! /usr/bin/env python3
#
# Copyright (c) 2017, Forschungszentrum Juelich GmbH
# Author: Yann Leprince <y.leprince@fz-juelich.de>
#
# This software is made available under the MIT licence, see LICENCE.txt.

import gzip
import re
import sys

import nibabel
import numpy as np
import pyvtk

# See description of OFF file format at
# http://www.geomview.org/docs/html/OFF.html

def off_mesh_file_to_vtk(input_filename, output_filename, data_format="binary",
                         coord_transform=None):
    """Convert a mesh file from OFF format to VTK format"""
    print("Reading {}".format(input_filename))
    with gzip.open(input_filename, "rt") as f:
        header_keyword = f.readline().strip()
        match = re.match(r"(ST)?(C)?(N)?(4)?(n)?OFF", header_keyword)
        # TODO check features from header keyword
        assert match
        assert not match.group(5)  # nOFF is unsupported
        dimension_line = f.readline().strip()
        match = re.match(r"([+-]?[0-9]+)\s+([+-]?[0-9]+)(\s+([+-]?[0-9]+))?",
                         dimension_line)
        assert match
        num_vertices = int(match.group(1))
        num_triangles = int(match.group(2))
        vertices = np.empty((num_vertices, 3), dtype=np.float)
        for i in range(num_vertices):
            components = f.readline().split()
            assert len(components) >= 3
            vertices[i, 0] = float(components[0])
            vertices[i, 1] = float(components[1])
            vertices[i, 2] = float(components[2])
        triangles = np.empty((num_triangles, 3), dtype=np.int_)
        for i in range(num_triangles):
            components = f.readline().split()
            assert len(components) >= 4
            assert components[0] == "3"
            triangles[i, 0] = float(components[1])
            triangles[i, 1] = float(components[2])
            triangles[i, 2] = float(components[3])
    print()
    print("{0} vertices and {1} triangles read"
          .format(num_vertices, num_triangles))

    points = vertices

    if coord_transform is not None:
        if coord_transform.shape[0] == 4:
            assert np.all(coord_transform[3, :] == [0, 0, 0, 1])
        points = points.T
        points = np.dot(coord_transform[:3, :3],points)
        points += coord_transform[:3, 3, np.newaxis]
        points = points.T
        if np.linalg.det(coord_transform[:3, :3]) < 0:
            # Flip the triangles to fix inside/outside
            triangles = np.flip(triangles, axis=1)

    # Gifti uses millimetres, Neuroglancer expects nanometres
    points *= 1e6

    # Workaround: dtype must be np.int_ (pyvtk does not recognize int32 as
    # integers)
    triangles = triangles.astype(np.int_)

    vtk_mesh = pyvtk.PolyData(points, polygons=triangles)

    vtk_data = pyvtk.VtkData(
        vtk_mesh,
        "Converted using "
        "https://github.com/HumanBrainProject/neuroglancer-scripts")
    vtk_data.tofile(output_filename, format=data_format)


def parse_command_line(argv):
    """Parse the script's command line."""
    import argparse
    parser = argparse.ArgumentParser(
        description="""\
Convert a mesh (readable by nibabel, e.g. in Gifti format) to VTK file format
""")
    parser.add_argument("input_filename")
    parser.add_argument("output_filename")
    parser.add_argument("--ascii", action="store_const",
                        dest="format", const="ascii", default="ascii",
                        help="save the VTK file in ASCII format (default)")
    parser.add_argument("--binary", action="store_const",
                        dest="format", const="binary",
                        help="save the VTK file in binary format"
                        " (not supported by Neuroglancer at this time)")
    parser.add_argument("--coord-transform",
                        help="affine transformation to be applied to the"
                        " coordinates, as a 4x4 matrix in homogeneous"
                        " coordinates, in comma-separated row-major order"
                        " (the last row is always 0 0 0 1 and may be omitted)"
                        " (e.g. --coord-transform=1,0,0,0,0,1,0,0,0,0,1,0)")
    args = parser.parse_args(argv[1:])

    if args.coord_transform is not None:
        try:
            matrix = np.fromstring(args.coord_transform, sep=",")
        except ValueError as exc:
            parser.error("cannot parse --coord-transform: {}"
                         .format(exc.args[0]))
        if len(matrix) == 12:
            matrix = matrix.reshape(3, 4)
        elif len(matrix) == 16:
            matrix = matrix.reshape(4, 4)
        else:
            parser.error("--coord-transform must have 12 or 16 elements"
                         " ({} passed)".format(len(matrix)))

        args.coord_transform = matrix

    return args


def main(argv):
    """The script's entry point."""
    args = parse_command_line(argv)
    return off_mesh_file_to_vtk(args.input_filename, args.output_filename,
                                data_format=args.format,
                                coord_transform=args.coord_transform) or 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
