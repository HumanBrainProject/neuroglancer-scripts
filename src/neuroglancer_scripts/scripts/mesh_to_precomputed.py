#! /usr/bin/env python3
#
# Copyright (c) 2017, Forschungszentrum Juelich GmbH
# Author: Yann Leprince <y.leprince@fz-juelich.de>
#
# This software is made available under the MIT licence, see LICENCE.txt.

import io
import logging
import pathlib
import sys

import nibabel
import numpy as np

import neuroglancer_scripts.accessor
import neuroglancer_scripts.mesh
import neuroglancer_scripts.precomputed_io as precomputed_io


logger = logging.getLogger(__name__)


def mesh_file_to_precomputed(input_path, dest_url, mesh_name=None,
                             mesh_dir=None, coord_transform=None, options={}):
    """Convert a mesh read by nibabel to Neuroglancer precomputed format"""
    input_path = pathlib.Path(input_path)
    accessor = neuroglancer_scripts.accessor.get_accessor_for_url(
        dest_url, options
    )
    info = precomputed_io.get_IO_for_existing_dataset(accessor).info
    if mesh_dir is None:
        mesh_dir = "mesh"  # default value
    if "mesh" not in info:
        info["mesh"] = mesh_dir
        # Write the updated info file
        precomputed_io.get_IO_for_new_dataset(
            info, accessor, overwrite_info=True
        )
    if mesh_dir != info["mesh"]:
        logger.critical("The provided --mesh-dir value does not match the "
                        "value stored in the info file")
        return 1
    if info["type"] != "segmentation":
        logger.warning('The dataset has type "image" instead of '
                       '"segmentation", Neuroglancer will not use the meshes.')

    if mesh_name is None:
        mesh_name = input_path.stem

    mesh = nibabel.load(str(input_path))

    points_list = mesh.get_arrays_from_intent("NIFTI_INTENT_POINTSET")
    assert len(points_list) == 1
    points = points_list[0].data

    triangles_list = mesh.get_arrays_from_intent("NIFTI_INTENT_TRIANGLE")
    assert len(triangles_list) == 1
    triangles = triangles_list[0].data

    if coord_transform is not None:
        points_dtype = points.dtype
        points, triangles = neuroglancer_scripts.mesh.affine_transform_mesh(
            points, triangles, coord_transform
        )
        # Convert vertices back to their original type to avoid the warning
        # that save_mesh_as_precomputed prints when downcasting to float32.
        points = points.astype(np.promote_types(points_dtype, np.float32),
                               casting="same_kind")

    # Gifti uses millimetres, Neuroglancer expects nanometres
    points *= 1e6

    io_buf = io.BytesIO()
    neuroglancer_scripts.mesh.save_mesh_as_precomputed(
        io_buf, points, triangles.astype("uint32")
    )
    accessor.store_file(mesh_dir + "/" + mesh_name, io_buf.getvalue(),
                        mime_type="application/octet-stream")


def parse_command_line(argv):
    """Parse the script's command line."""
    import argparse
    parser = argparse.ArgumentParser(
        description="""\
Convert a mesh to Neuroglancer pre-computed mesh format.

This tool can convert any mesh format that is readable by nibabel, e.g. Gifti.

The resulting files are so-called mesh fragments, which will not be directly
visible by Neuroglancer. The fragments need to be linked to the integer labels
of the associated segmentation image, through small JSON files in the mesh
directory (see the link-mesh-fragments command).
""")
    parser.add_argument("input_mesh", type=pathlib.Path,
                        help="input mesh file to be read by Nibabel")
    parser.add_argument("dest_url",
                        help="base directory/URL of the output dataset")

    parser.add_argument("--mesh-dir", default=None,
                        help='sub-directory of the dataset where the mesh '
                        'file(s) will be written. If given, this value must '
                        'match the "mesh" key of the info file. It will be '
                        'written to the info file if not already present. '
                        '(default: "mesh").')
    parser.add_argument("--mesh-name", default=None,
                        help="name of the precomputed mesh file (default: "
                        "basename of the input mesh file)")
    parser.add_argument("--coord-transform",
                        help="affine transformation to be applied to the"
                        " coordinates, as a 4x4 matrix in homogeneous"
                        " coordinates, with the translation in millimetres,"
                        " in comma-separated row-major order"
                        " (the last row is always 0 0 0 1 and may be omitted)"
                        " (e.g. --coord-transform=1,0,0,0,0,1,0,0,0,0,1,0)")

    neuroglancer_scripts.accessor.add_argparse_options(parser,
                                                       read=True, write=True)

    args = parser.parse_args(argv[1:])
    # TODO factor in library
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


def main(argv=sys.argv):
    """The script's entry point."""
    import neuroglancer_scripts.utils
    neuroglancer_scripts.utils.init_logging_for_cmdline()
    args = parse_command_line(argv)
    return mesh_file_to_precomputed(args.input_mesh, args.dest_url,
                                    mesh_name=args.mesh_name,
                                    mesh_dir=args.mesh_dir,
                                    coord_transform=args.coord_transform,
                                    options=vars(args)) or 0


if __name__ == "__main__":
    sys.exit(main())
