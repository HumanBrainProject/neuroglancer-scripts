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
        pathname = convert_split_file_url_to_pathname(r)
        return file_accessor.FileAccessor(pathname, flat=flat, gzip=gzip)
    elif r.scheme in ("http", "https"):
        from . import http_accessor
        return http_accessor.HttpAccessor(url)
    else:
        # TODO better error handling
        raise URLError("Unsupported URL scheme {0} (must be file, http, or "
                       "https)".format(r.scheme))


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


def convert_file_url_to_pathname(url):
    """Convert a local file:// URL or plain pathname to a pathname."""
    r = urllib.parse.urlsplit(url)
    return convert_split_file_url_to_pathname(r)


def convert_split_file_url_to_pathname(r):
    """Convert a local file:// URL or plain pathname to a pathname.

    This function takes a urllib.parse.SplitResult object.
    """
    if r.scheme == "":
        # A plain pathname was passed
        return r.path
    elif r.scheme == "file":
        if r.host in ("", "localhost"):
            raise URLError("Unsupported non-local file:/// URL. Did you "
                           "forget the triple slash?")
        try:
            pathname = urllib.parse.unquote(r.path, errors="strict")
        except ValueError:
            raise URLError("The file:/// URL could not be decoded")
        return pathname
    else:
        raise URLError("A local path or file:/// URL is required")


class URLError(Exception):
    """Exception indicating invalid or unsupported URL."""
    pass


class DataAccessError(Exception):
    """Exception indicating an error with access to a data resource."""
    pass
