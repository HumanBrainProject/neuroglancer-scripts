#! /usr/bin/env python3
#
# Copyright (c) 2017, Forschungszentrum Juelich GmbH
# Author: Pavel Chervakov <p.chervakov@fz-juelich.de>
#
# This software is made available under the MIT licence, see LICENCE.txt.

# noqa

"""
Convert a mesh from STL ASCII to Neuroglancer pre-computed mesh format

Currently STL triangles are just written to the output as is, i.e. normals are
not considered and equal vertices are not reused.
"""

import gzip
import struct
import sys
from functools import partial

# import numpy as np

__VERTEX_STR_PREFIX = '   vertex '


def __get_vertex(vstr: str, voxel_size):
    assert vstr.startswith(__VERTEX_STR_PREFIX)
    return list(map(lambda v: float(v) * 1e6 * voxel_size,
                    vstr[len(__VERTEX_STR_PREFIX):-1].split()))


def __get_vertices(septuple, voxel_size):
    assert len(septuple) == 7
    assert septuple[0].startswith(' facet normal')
    assert septuple[1] == '  outer loop\n'
    assert septuple[5] == '  endloop\n'
    assert septuple[6] == ' endfacet\n'
    return [__get_vertex(septuple[2], voxel_size), __get_vertex(
        septuple[3], voxel_size), __get_vertex(septuple[4], voxel_size)]


def stl_file_to_precomputed(
        input_filename, output_filename, voxel_size=1.0, compress=True):
    with open(input_filename) as input_file:
        lines = input_file.readlines()
    assert lines[0] == 'solid ascii\n'
    assert lines[-1] == 'endsolid\n'
    assert (len(lines) - 2) % 7 == 0

    gv = partial(__get_vertices, voxel_size=voxel_size)
    triples = list(map(gv, [lines[(i * 7) + 1: ((i + 1) * 7) + 1]
                            for i in range((len(lines) - 2) // 7)]))
    vertices = [vertex for triple in triples for vertex in triple]
    num_vertices = len(vertices)
    buf = bytearray()
    buf += struct.pack("<I", num_vertices)
    for vertex in vertices:
        buf += struct.pack("<fff", vertex[0], vertex[1], vertex[2])

    assert len(buf) == 4 + 4 * 3 * num_vertices

    for i in range(num_vertices):
        buf += struct.pack("<I", i)

    if compress:
        with gzip.open(output_filename + ".gz", "wb") as output_file:
            output_file.write(bytes(buf))
    else:
        with open(output_filename, "wb") as output_file:
            output_file.write(bytes(buf))

    if __name__ == "__main__":
        print('done')


def parse_command_line(argv):
    """Parse the script's command line."""
    import argparse
    parser = argparse.ArgumentParser(
        description="Convert a mesh from STL ASCII to Neuroglancer "
                    "pre-computed mesh format")
    parser.add_argument("input_filename")
    parser.add_argument("output_filename")
    parser.add_argument("--voxel-size", help="Voxel size in mm. Only "
                        "isotropic voxels are supported for now. Default is "
                        "1.0",
                        type=float, default=1.0)
    parser.add_argument("--no-compression", help="Don't gzip the output.",
                        action="store_false", dest="compress")
    args = parser.parse_args(argv[1:])
    return args


def main(argv):
    """The script's entry point."""
    args = parse_command_line(argv)
    return stl_file_to_precomputed(
        args.input_filename, args.output_filename, args.voxel_size,
        args.compress) or 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
