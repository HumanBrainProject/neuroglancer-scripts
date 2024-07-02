# Copyright (c) 2018, 2023 Forschungszentrum Juelich GmbH
#
# Author: Yann Leprince <y.leprince@fz-juelich.de>
# Author: Xiao Gui <xgui3783@gmail.com>
#
# This software is made available under the MIT licence, see LICENCE.txt.

"""Low-level file access to Neuroglancer pre-computed datasets.

The central component here is the :class:`Accessor` base class. Use
:func:`get_accessor_for_url` for instantiating a concrete accessor object.
"""

import json
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
        from neuroglancer_scripts import file_accessor, sharded_base
        flat = accessor_options.get("flat", False)
        gzip = accessor_options.get("gzip", True)
        compresslevel = accessor_options.get("compresslevel", 9)
        pathname = _convert_split_file_url_to_pathname(r)

        accessor = file_accessor.FileAccessor(pathname, flat=flat, gzip=gzip,
                                              compresslevel=compresslevel)
        is_sharding = False
        if accessor_options.get("sharding"):
            is_sharding = True
        if not is_sharding:
            try:
                info = json.loads(accessor.fetch_file("info"))
                if sharded_base.ShardedAccessorBase.info_is_sharded(info):
                    is_sharding = True
            except (DataAccessError, json.JSONDecodeError):
                # In the event that info does not exist
                # Or info is malformed
                # Fallback to default behavior
                ...

        if is_sharding:
            from neuroglancer_scripts import sharded_file_accessor
            return sharded_file_accessor.ShardedFileAccessor(pathname)

        return accessor

    elif r.scheme in ("http", "https"):
        from neuroglancer_scripts import http_accessor, sharded_base
        accessor = http_accessor.HttpAccessor(url)

        is_sharding = False
        if "sharding" in accessor_options:
            is_sharding = True
        if not is_sharding:
            try:
                info = json.loads(accessor.fetch_file("info"))
                if sharded_base.ShardedAccessorBase.info_is_sharded(info):
                    is_sharding = True
            except (DataAccessError, json.JSONDecodeError):
                # In the event that info does not exist
                # Or info is malformed
                # Fallback to default behavior
                ...

        if is_sharding:
            from neuroglancer_scripts import sharded_http_accessor
            return sharded_http_accessor.ShardedHttpAccessor(url)
        return accessor
    else:
        raise URLError(f"Unsupported URL scheme {r.scheme} (must be file, "
                       "http, or https)")


def add_argparse_options(parser, write_chunks=True, write_files=True):
    """Add command-line options for file access.

    :param argparse.ArgumentParser parser: an argument parser
    :param bool write_chunks: whether to add options for chunk writing
    :param bool write_files: whether to add options for file writing

    The accesor options can be obtained from command-line arguments with
    :func:`add_argparse_options` and passed to :func:`get_accessor_for_url`::

        import argparse
        parser = argparse.ArgumentParser()
        add_argparse_options(parser)
        args = parser.parse_args()
        get_accessor_for_url(url, vars(args))
    """
    if write_chunks or write_files:
        group = parser.add_argument_group("Options for file storage")
        group.add_argument("--no-gzip", "--no-compression",
                           action="store_false", dest="gzip",
                           help="Don't gzip the output.")
        group.add_argument("--compresslevel", type=int, default=9,
                           choices=range(0, 10),
                           help="Gzip compression level (0-9, default 9)")
    if write_chunks:
        group.add_argument(
            "--flat", action="store_true",
            help="Store all chunks for each resolution with a flat layout, as "
            "Neuroglancer expects. By default the chunks are stored in "
            "sub-directories, which requires a specially configured web "
            "server (see https://github.com/HumanBrainProject/neuroglancer-"
            "docker). " "Do not use this option for large images, or you risk "
            "running into problems with directories containing huge numbers "
            "of files.")


class Accessor:
    """Access a Neuroglancer pre-computed pyramid.

    An accessor encapsulates access to the files that form a Neuroglancer
    pre-computed dataset. It works with sequences of bytes, without
    interpreting the file contents.

    You can inherit from this class in order to implement a new accessor (see
    :class:`~neuroglancer_scripts.file_accessor.FileAccessor`,
    :class:`~neuroglancer_scripts.http_accessor.HttpAccessor`).
    """

    can_read = False
    """This accessor is able to read data."""

    can_write = False
    """This accessor is able to write data."""

    def file_exists(self, relative_path):
        """Test existence of a file relative to the base directory.

        :param str relative_path: path to the file relative to the base
                                  directory of the pyramid
        :returns: True if the file exists
        :rtype: bool
        :raises DataAccessError: if an error occurs when probing file existence
        :raises NotImplementedError: if :attr:`can_read` is False
        """
        raise NotImplementedError

    def fetch_file(self, relative_path):
        """Fetch a file relative to the precomputed pyramid's base directory.

        :param str relative_path: path to the file relative to the base
                                  directory of the pyramid
        :returns: contents of the fetched file
        :rtype: bytes
        :raises DataAccessError: if the requested file cannot be retrieved
        :raises NotImplementedError: if :attr:`can_read` is False
        """
        raise NotImplementedError

    def store_file(self, relative_path, buf,
                   mime_type="application/octet-stream",
                   overwrite=False):
        """Store a file relative to the precomputed pyramid's base directory.

        :param str relative_path: path to the file relative to the base
                                  directory of the pyramid
        :param bytes buf: the contents of the file to be stored
        :param str mime_type: MIME type of the file
        :param bool overwrite: whether to allow overwriting an existing file
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

    def store_chunk(self, buf, key, chunk_coords,
                    mime_type="application/octet-stream",
                    overwrite=False):
        """Store a chunk in the pyramid from a bytes buffer.

        :param str key: the scale's key
        :param tuple chunk_coords: tuple of the chunk coordinates (xmin, xmax,
                                   ymin, ymax, zmin, zmax)
        :param str mime_type: MIME type of the chunk data
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
