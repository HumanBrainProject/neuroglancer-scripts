#! /usr/bin/env python3
#
# Copyright (c) 2018, CEA
# Author: Yann Leprince <yann.leprince@cea.fr>
#
# This software is made available under the MIT licence, see LICENCE.txt.

import csv
import json
import os.path
import sys


def fragment_exists(fragment_name, mesh_dir):
    return (
        os.path.isfile(os.path.join(mesh_dir, fragment_name))
        or os.path.isfile(os.path.join(mesh_dir, fragment_name + ".gz"))
    )


def make_mesh_fragment_links(input_csv, output_mesh_dir):
    with open(input_csv, newline="") as csv_file:
        csv_reader = csv.reader(csv_file)
        for line in csv_reader:
            numeric_label = int(line[0])
            fragment_list = line[1:]
            if False:  # TODO make filtering optional
                fragment_list = [f for f in fragment_list
                                 if fragment_exists(f, output_mesh_dir)]
            else:
                for fragment_name in fragment_list:
                    if not fragment_exists(fragment_name, output_mesh_dir):
                        print("WARNING: missing fragment {0}"
                              .format(fragment_name))
                # TODO output a warning for missing fragments
            fragment_json_filename = os.path.join(output_mesh_dir,
                                                  "{0}:0".format(numeric_label))
            # TODO use accessor
            with open(fragment_json_filename, "w") as fragment_json_file:
                json.dump({"fragments": fragment_list}, fragment_json_file,
                          separators=(",", ":"))


def parse_command_line(argv):
    """Parse the script's command line."""
    import argparse
    parser = argparse.ArgumentParser(
        description="""\
Create JSON files linking pre-computed mesh fragments to labels of a segmentation.
""")
    parser.add_argument("input_csv")
    parser.add_argument("output_mesh_dir")
    args = parser.parse_args(argv[1:])
    return args


def main(argv=sys.argv):
    """The script's entry point."""
    args = parse_command_line(argv)
    return make_mesh_fragment_links(args.input_csv, args.output_mesh_dir) or 0


if __name__ == "__main__":
    sys.exit(main())
