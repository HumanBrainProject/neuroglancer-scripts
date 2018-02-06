# Copyright (c) 2018 Forschungszentrum Juelich GmbH
# Author: Yann Leprince <y.leprince@fz-juelich.de>
#
# This software is made available under the MIT licence, see LICENCE.txt.

import urllib.parse

__all__ = [
    "get_accessor_for_url",
    "add_argparse_options",
    "Accessor",
    "DataAccessError",
    "convert_file_url_to_pathname",
    "URLError",
]

_CHUNK_PATTERN_FLAT = "{key}/{0}-{1}_{2}-{3}_{4}-{5}"

def get_accessor_for_url(url, accessor_options={}):
    """Create an accessor object from a URL or pathname.

    :param str url: URL or plain local pathname to the pyramid
    :param dict accessor_options: options passed to the accessor as kwargs.
    :returns: an instance of a sub-class of :class:`Accessor`
    :rtype: Accessor
    """
    url = _strip_precomputed(url)
    r = urllib.parse.urlsplit(url)
    if r.scheme in ("", "file"):
        from neuroglancer_scripts import file_accessor
        flat = accessor_options.get("flat", False)
        gzip = accessor_options.get("gzip", True)
        pathname = _convert_split_file_url_to_pathname(r)
        return file_accessor.FileAccessor(pathname, flat=flat, gzip=gzip)
    elif r.scheme in ("http", "https"):
        from neuroglancer_scripts import http_accessor
        return http_accessor.HttpAccessor(url)
    else:
        raise URLError("Unsupported URL scheme {0} (must be file, http, or "
                       "https)".format(r.scheme))


def add_argparse_options(parser, read=True, write=True):
    """Add command-line options for file access.

    :param argparse.ArgumentParser parser: an argument parser
    :param bool read: whether to add options for file reading
    :param bool write: whether to add options for file writing

    The accesor options can be obtained from command-line arguments with
    :func:`add_argparse_options` and passed to :func:`get_accessor_for_url`::

        import argparse
        parser = argparse.ArgumentParser()
        add_argparse_options(parser)
        args = parser.parse_args()
        get_accessor_for_url(url, vars(args))
    """
    if write:
        group = parser.add_argument_group("Options for file storage")
        group.add_argument(
            "--flat", action="store_true",
            help="Store all chunks for each resolution with a flat layout, as "
            "Neuroglancer expects. By default the chunks are stored in "
            "sub-directories, which requires a specially configured web "
            "server (see https://github.com/HumanBrainProject/neuroglancer-"
            "docker). " "Do not use this option for large images, or you risk "
            "running into problems with directories containing huge numbers "
            "of files.")
        group.add_argument("--no-gzip", "--no-compression",
                           action="store_false", dest="gzip",
                           help="Don't gzip the output.")


class Accessor:
    """Access a Neuroglancer pre-computed pyramid.

    An accessor encapsulates access to the files that form a Neuroglancer
    pre-computed dataset. It works with sequences of bytes, without
    interpreting the file contents.

    .. todo:: add I/O for auxiliary files (e.g. mesh fragments)

    You can inherit from this class in order to implement a new accessor (see
    :class:`~neuroglancer_scripts.file_accessor.FileAccessor`,
    :class:`~neuroglancer_scripts.http_accessor.HttpAccessor`).
    """

    can_read = False
    """This accessor is able to read data."""

    can_write = False
    """This accessor is able to write data."""

    def fetch_info(self):
        """Fetch the pyramid's JSON *info* file.

        :returns: the parsed *info* file
        :rtype: dict
        :raises DataAccessError: if the *info* file cannot be retrieved
        :raises NotImplementedError: if :attr:`can_read` is False
        """
        raise NotImplementedError

    def store_info(self, info):
        """Store the pyramid's JSON *info* file.

        :param dict info: the *info* to be stored
        :raises DataAccessError: if the *info* file cannot be retrieved
        :raises NotImplementedError: if :attr:`can_write` is False
        """
        raise NotImplementedError

    def fetch_chunk(self, key, chunk_coords):
        """Fetch a chunk from the pyramid as a bytes buffer.

        :param str key: the scale's key
        :param tuple chunk_coords: tuple of the chunk coordinates (xmin, xmax,
                                   ymin, ymax, zmin, zmax)
        :returns: the data for the chunk (uncompressed for gzip)
        :rtype: bytes
        :raises DataAccessError: if the chunk cannot be retrieved
        :raises NotImplementedError: if :attr:`can_read` is False
        """
        raise NotImplementedError

    def store_chunk(self, buf, key, chunk_coords, already_compressed=False):
        """Store a chunk in the pyramid from a bytes buffer.

        :param str key: the scale's key
        :param tuple chunk_coords: tuple of the chunk coordinates (xmin, xmax,
                                   ymin, ymax, zmin, zmax)
        :param bool already_compressed: do not compress the chunk (ignore the
                                        accessor's gzip option)
        :raises DataAccessError: if the chunk cannot be stored
        :raises NotImplementedError: if :attr:`can_write` is False
        """
        raise NotImplementedError


class DataAccessError(Exception):
    """Exception indicating an error with access to a data resource."""
    # TODO check that passing a message and an (OSError) exception prints
    # correctly
    pass


def convert_file_url_to_pathname(url):
    """Convert a local file:// URL or plain pathname to a pathname.

    :param str url: the URL or pathname to convert into a local pathname
    :returns: a local pathname suitable for :func:`open`
    :rtype: str
    :raises URLError: if the provided string looks like a URL but is not a
                      valid local file:/// URL
    """
    url = _strip_precomputed(url)
    r = urllib.parse.urlsplit(url)
    return _convert_split_file_url_to_pathname(r)


class URLError(Exception):
    """Exception indicating an invalid or unsupported URL."""
    pass


def _convert_split_file_url_to_pathname(r):
    if r.scheme == "":
        # A plain pathname was passed
        return r.path
    elif r.scheme == "file":
        if r.netloc not in ("", "localhost"):
            raise URLError("Unsupported non-local file:/// URL. Did you "
                           "forget the triple slash?")
        try:
            pathname = urllib.parse.unquote(r.path, errors="strict")
        except ValueError as exc:
            raise URLError("The file:/// URL could not be decoded") from exc
        return pathname
    else:
        raise URLError("A local path or file:/// URL is required")


def _strip_precomputed(url):
    if url.startswith("precomputed://"):
        url = url[len("precomputed://"):]
    return url
