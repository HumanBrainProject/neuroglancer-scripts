#! /usr/bin/env python3
#
# Copyright (c) 2017, Forschungszentrum Juelich GmbH
# Author: Yann Leprince <y.leprince@fz-juelich.de>
#
# This software is made available under the MIT licence, see LICENCE.txt.

import gzip
import struct
import sys

import nibabel
import numpy as np


def progress(text):
    sys.stderr.write(text)
    sys.stderr.flush()


def mesh_file_to_precomputed(input_filename, output_filename,
                             coord_transform=None):
    """Convert a mesh read by nibabel to Neuroglancer precomputed format"""
    print("Reading {}".format(input_filename))
    mesh = nibabel.load(input_filename)
    print()
    print("Summary")
    print("=======")
    mesh.print_summary()

    points_list = mesh.get_arrays_from_intent("NIFTI_INTENT_POINTSET")
    assert len(points_list) == 1
    points = points_list[0].data

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

    triangles_list = mesh.get_arrays_from_intent("NIFTI_INTENT_TRIANGLE")
    assert len(triangles_list) == 1
    triangles = triangles_list[0].data

    # Gifti uses millimetres, Neuroglancer expects nanometres
    points *= 1e6

    num_vertices = len(points)

    buf = bytearray()
    buf += struct.pack("<I", num_vertices)
    progress("Preparing vertices... ")
    for vertex in points:
        buf += struct.pack("<fff", vertex[0], vertex[1], vertex[2])
    assert len(buf) == 4 + 4 * 3 * num_vertices
    progress("Preparing triangles... ")
    for triangle in triangles:
        buf += struct.pack("<III", *triangle)
    progress("done.\nWriting file...")
    with gzip.open(output_filename + ".gz", "wb") as output_file:
        output_file.write(bytes(buf))
    progress("done.\n")


def parse_command_line(argv):
    """Parse the script's command line."""
    import argparse
    parser = argparse.ArgumentParser(
        description="""\
Convert a mesh (readable by nibabel, e.g. in Gifti format) to Neuroglancer
pre-computed mesh format
""")
    parser.add_argument("input_filename")
    parser.add_argument("output_filename")
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
    return mesh_file_to_precomputed(args.input_filename, args.output_filename,
                                    coord_transform=args.coord_transform) or 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
