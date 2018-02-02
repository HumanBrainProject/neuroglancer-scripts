# Copyright (c) 2018 Forschungszentrum Juelich GmbH
# Author: Yann Leprince <y.leprince@fz-juelich.de>
#
# This software is made available under the MIT licence, see LICENCE.txt.

import urllib.parse

import requests


from .accessor import CHUNK_PATTERN_FLAT


class HttpAccessor:
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
        url_suffix = CHUNK_PATTERN_FLAT.format(
            xmin, xmax, ymin, ymax, zmin, zmax, key=key)
        return self.base_url + url_suffix

    def fetch_chunk(self, key, chunk_coords):
        chunk_url = self.chunk_url(key, chunk_coords)
        r = requests.get(chunk_url)
        r.raise_for_status()
        return r.content

    def fetch_info(self):
        info_url = self.base_url + "info"
        r = requests.get(info_url)
        r.raise_for_status()
        return r.json()
