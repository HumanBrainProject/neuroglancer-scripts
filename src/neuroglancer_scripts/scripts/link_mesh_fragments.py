#! /usr/bin/env python3
#
# Copyright (c) 2018, CEA
# Author: Yann Leprince <yann.leprince@cea.fr>
#
# This software is made available under the MIT licence, see LICENCE.txt.

import csv
import json
import logging
import os.path
import sys

import neuroglancer_scripts.accessor
import neuroglancer_scripts.precomputed_io as precomputed_io

logger = logging.getLogger(__name__)


def fragment_exists(fragment_name, mesh_dir):
    return (
        os.path.isfile(os.path.join(mesh_dir, fragment_name))
        or os.path.isfile(os.path.join(mesh_dir, fragment_name + ".gz"))
    )


def make_mesh_fragment_links(input_csv, dest_url, no_colon_suffix=False,
                             options={}):
    if no_colon_suffix:
        filename_format = "{0}/{1}"
    else:
        filename_format = "{0}/{1}:0"
    accessor = neuroglancer_scripts.accessor.get_accessor_for_url(
        dest_url, options
    )
    info = precomputed_io.get_IO_for_existing_dataset(accessor).info
    if "mesh" not in info:
        logger.critical('The info file is missing the "mesh" key, please '
                        'use mesh-to-precomputed first.')
        return 1
    mesh_dir = info["mesh"]

    with open(input_csv, newline="") as csv_file:
        for line in csv.reader(csv_file):
            numeric_label = int(line[0])
            fragment_list = line[1:]
            # Output a warning for missing fragments
            for fragment_name in fragment_list:
                if not accessor.file_exists(mesh_dir + "/" + fragment_name):
                    logger.warning("missing fragment %s", fragment_name)
            relative_filename = filename_format.format(mesh_dir, numeric_label)
            json_str = json.dumps({"fragments": fragment_list},
                                  separators=(",", ":"))
            accessor.store_file(relative_filename, json_str.encode("utf-8"),
                                mime_type="application/json")


def parse_command_line(argv):
    """Parse the script's command line."""
    import argparse
    parser = argparse.ArgumentParser(
        description="""\
Create JSON files linking mesh fragments to labels of a segmentation layer.

The input is a CSV file, where each line contains the integer label in the
first cell, followed by an arbitrary number of cells whose contents name the
fragment files corresponding to the given integer label.
""")
    parser.add_argument("input_csv",
                        help="CSV file containing the mapping between integer "
                        "labels and mesh fragment name, in the format "
                        "described above")
    parser.add_argument("dest_url",
                        help="base directory/URL of the output dataset")

    parser.add_argument("--no-colon-suffix",
                        dest="no_colon_suffix", action="store_true",
                        help="omit the :0 suffix in the name of created JSON "
                        "files (e.g. 10 instead of 10:0). This is necessary "
                        "on filesystems that disallow colons, such as FAT.")

    neuroglancer_scripts.accessor.add_argparse_options(parser,
                                                       write_chunks=False)

    args = parser.parse_args(argv[1:])
    return args


def main(argv=sys.argv):
    """The script's entry point."""
    import neuroglancer_scripts.utils
    neuroglancer_scripts.utils.init_logging_for_cmdline()
    args = parse_command_line(argv)
    return make_mesh_fragment_links(args.input_csv, args.dest_url,
                                    no_colon_suffix=args.no_colon_suffix,
                                    options=vars(args)) or 0


if __name__ == "__main__":
    sys.exit(main())
