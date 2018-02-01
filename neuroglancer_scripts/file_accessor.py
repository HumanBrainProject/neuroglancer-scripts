# Copyright (c) 2016, 2017, 2018 Forschungszentrum Juelich GmbH
# Author: Yann Leprince <y.leprince@fz-juelich.de>
#
# This software is made available under the MIT licence, see LICENCE.txt.

import gzip
import json
import os.path

from .accessor import CHUNK_PATTERN_FLAT

CHUNK_PATTERN_SUBDIR = "{key}/{0}-{1}/{2}-{3}/{4}-{5}"


class FileAccessor:
    def __init__(self, base_dir=".", flat=False, gzip=True):
        self.base_dir = base_dir
        if flat:
            self.chunk_pattern = CHUNK_PATTERN_FLAT
        else:
            self.chunk_pattern = CHUNK_PATTERN_SUBDIR
        self.gzip = gzip

    def fetch_info(self):
        with open(os.path.join(self.base_dir, "info")) as f:
            return json.load(f)

    def store_info(self, info):
        os.makedirs(self.base_dir, exist_ok=True)
        with open(os.path.join(self.base_dir, "info"), "w") as f:
            json.dump(info, f, separators=(",", ":"), sort_keys=True)

    def chunk_filename(self, key, chunk_coords):
        xmin, xmax, ymin, ymax, zmin, zmax = chunk_coords
        chunk_filename = self.chunk_pattern.format(
            xmin, xmax, ymin, ymax, zmin, zmax, key=key)
        return os.path.join(self.base_dir, chunk_filename)

    def fetch_chunk(self, key, chunk_coords):
        chunk_path = self.chunk_filename(key, chunk_coords)
        try:
            f = gzip.open(chunk_path + ".gz", "rb")
        except OSError:
            f = open(chunk_path, "rb")
        with f:
            return f.read()

    def store_chunk(self, buf, key, chunk_coords, already_compressed):
        chunk_path = self.chunk_filename(key, chunk_coords)
        os.makedirs(os.path.dirname(chunk_path), exist_ok=True)
        if self.gzip and not already_compressed:
            with gzip.open(chunk_path + ".gz", "wb") as f:
                f.write(buf)
        else:
            with open(chunk_path, "wb") as f:
                f.write(buf)
