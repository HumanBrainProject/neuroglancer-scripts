# Copyright (c) 2018 Forschungszentrum Juelich GmbH
# Author: Yann Leprince <y.leprince@fz-juelich.de>
#
# This software is made available under the MIT licence, see LICENCE.txt.

"""Access to a Neuroglancer pre-computed dataset over HTTP.

See the :mod:`~neuroglancer_scripts.accessor` module for a description of the
API.
"""

import urllib.parse

import requests

import neuroglancer_scripts.accessor
from neuroglancer_scripts.accessor import _CHUNK_PATTERN_FLAT, DataAccessError

__all__ = [
    "HttpAccessor",
]


class HttpAccessor(neuroglancer_scripts.accessor.Accessor):
    """Access a Neuroglancer pre-computed pyramid with HTTP.

    .. note::
       This is a read-only accessor.

    :param str base_url: the URL containing the pyramid
    """

    can_read = True
    can_write = False

    def __init__(self, base_url):
        self._session = requests.Session()

        # Fix the base URL to end with a slash, discard query and fragment
        r = urllib.parse.urlsplit(base_url)
        self.base_url = urllib.parse.urlunsplit((
            r.scheme, r.netloc,
            r.path if r.path[-1] == "/" else r.path + "/",
            "", ""))

    def chunk_relative_url(self, key, chunk_coords):
        xmin, xmax, ymin, ymax, zmin, zmax = chunk_coords
        url_suffix = _CHUNK_PATTERN_FLAT.format(
            xmin, xmax, ymin, ymax, zmin, zmax, key=key)
        return url_suffix

    def fetch_chunk(self, key, chunk_coords):
        chunk_url = self.chunk_relative_url(key, chunk_coords)
        return self.fetch_file(chunk_url)

    def file_exists(self, relative_path):
        file_url = self.base_url + relative_path
        try:
            r = self._session.head(file_url)
            if r.status_code == requests.codes.not_found:
                return False
            r.raise_for_status()
        except requests.exceptions.RequestException as exc:
            raise DataAccessError("Error probing the existence of "
                                  f"{file_url}: {exc}") from exc
        return True

    def fetch_file(self, relative_path):
        file_url = self.base_url + relative_path
        try:
            r = self._session.get(file_url)
            r.raise_for_status()
        except requests.exceptions.RequestException as exc:
            raise DataAccessError(f"Error reading {file_url}: {exc}") from exc
        return r.content
