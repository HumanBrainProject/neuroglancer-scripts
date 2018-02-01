# Copyright (c) 2018 Forschungszentrum Juelich GmbH
# Author: Yann Leprince <y.leprince@fz-juelich.de>
#
# This software is made available under the MIT licence, see LICENCE.txt.

import urllib.parse

CHUNK_PATTERN_FLAT = "{key}/{0}-{1}_{2}-{3}_{4}-{5}"

def get_accessor_for_url(url, accessor_options={}):
    if url.startswith("precomputed://"):
        url = url[len("precomputed://"):]
    r = urllib.parse.urlsplit(url)
    if r.scheme in ("", "file"):
        from . import file_accessor
        flat = accessor_options.get("flat", False)
        gzip = accessor_options.get("gzip", True)
        return file_accessor.FileAccessor(url, flat=flat, gzip=gzip)
    elif r.scheme in ("http", "https"):
        from . import http_accessor
        return http_accessor.HttpAccessor(url)
    else:
        # TODO better error handling
        raise ValueError("unsupported scheme {0}".format(r.scheme))

def add_argparse_options(parser):
    group = parser.add_argument_group("Options for file storage")
    group.add_argument(
        "--flat", action="store_true",
        help="Store all chunks for each resolution with a flat layout, as "
        "Neuroglancer expects. By default the chunks are stored in "
        "sub-directories, which requires a specially configured web server "
        "(see https://github.com/HumanBrainProject/neuroglancer-docker). "
        "Do not use this option for large images, or you risk running into "
        "problems with directories containing huge numbers of files.")
    group.add_argument("--no-gzip", "--no-compression",
                       action="store_false", dest="gzip",
                       help="Don't gzip the output.")
