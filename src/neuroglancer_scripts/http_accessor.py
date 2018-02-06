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


# TODO DataAccessError
class HttpAccessor(neuroglancer_scripts.accessor.Accessor):
    """Access a Neuroglancer pre-computed pyramid with HTTP.

    .. note::
       This is a read-only accessor.

    :param str base_url: the URL containing the pyramid
    """

    can_read = True
    can_write = False

    def __init__(self, base_url):
        # Fix the base URL to end with a slash, discard query and fragment
        r = urllib.parse.urlsplit(base_url)
        self.base_url = urllib.parse.urlunsplit((
            r.scheme, r.netloc,
            r.path if r.path[-1] == "/" else r.path + "/",
            "", ""))

    def chunk_url(self, key, chunk_coords):
        xmin, xmax, ymin, ymax, zmin, zmax = chunk_coords
        url_suffix = _CHUNK_PATTERN_FLAT.format(
            xmin, xmax, ymin, ymax, zmin, zmax, key=key)
        return self.base_url + url_suffix

    def fetch_chunk(self, key, chunk_coords):
        chunk_url = self.chunk_url(key, chunk_coords)
        try:
            r = requests.get(chunk_url)
            r.raise_for_status()
        except requests.exceptions.RequestException as exc:
            raise DataAccessError("Error reading chunk from {0}:"
                                  .format(chunk_url, exc)) from exc
        return r.content

    def fetch_info(self):
        info_url = self.base_url + "info"
        try:
            r = requests.get(info_url)
            r.raise_for_status()
        except requests.exceptions.RequestException as exc:
            raise DataAccessError("Error reading {0}: {1}"
                                  .format(info_url, exc)) from exc
        return r.json()
