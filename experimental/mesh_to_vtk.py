#! /usr/bin/env python3
#
# Copyright (c) 2017, Forschungszentrum Juelich GmbH
# Author: Yann Leprince <y.leprince@fz-juelich.de>
#
# This software is made available under the MIT licence, see LICENCE.txt.

import sys

import nibabel
import numpy as np

import neuroglancer_scripts.mesh


def mesh_file_to_vtk(input_filename, output_filename, data_format="ascii",
                     coord_transform=None):
    """Convert a mesh file read by nibabel to VTK format"""
    print("Reading {}".format(input_filename))
    mesh = nibabel.load(input_filename)
    print()
    print("Summary")
    print("=======")
    mesh.print_summary()

    points_list = mesh.get_arrays_from_intent("NIFTI_INTENT_POINTSET")
    assert len(points_list) == 1
    points = points_list[0].data

    triangles_list = mesh.get_arrays_from_intent("NIFTI_INTENT_TRIANGLE")
    assert len(triangles_list) == 1
    triangles = triangles_list[0].data

    if coord_transform is not None:
        if coord_transform.shape[0] == 4:
            assert np.all(coord_transform[3, :] == [0, 0, 0, 1])
        points = points.T
        points = np.dot(coord_transform[:3, :3], points)
        points += coord_transform[:3, 3, np.newaxis]
        points = points.T
        if np.linalg.det(coord_transform[:3, :3]) < 0:
            # Flip the triangles to fix inside/outside
            triangles = np.flip(triangles, axis=1)

    # Gifti uses millimetres, Neuroglancer expects nanometres
    points *= 1e6

    with open(output_filename, "wt") as output_file:
        neuroglancer_scripts.mesh.save_mesh_as_neuroglancer_vtk(
            output_file, points, triangles
        )


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
    return mesh_file_to_vtk(args.input_filename, args.output_filename,
                            data_format=args.format,
                            coord_transform=args.coord_transform) or 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
