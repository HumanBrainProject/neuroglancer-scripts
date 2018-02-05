# Copyright (c) 2016, 2017, 2018 Forschungszentrum Juelich GmbH
# Author: Yann Leprince <y.leprince@fz-juelich.de>
#
# This software is made available under the MIT licence, see LICENCE.txt.

import gzip
import json
import os.path

import neuroglancer_scripts.accessor
from neuroglancer_scripts.accessor import _CHUNK_PATTERN_FLAT, DataAccessError


__all__ = [
    "FileAccessor",
]


_CHUNK_PATTERN_SUBDIR = "{key}/{0}-{1}/{2}-{3}/{4}-{5}"


class FileAccessor(neuroglancer_scripts.accessor.Accessor):
    """Access a Neuroglancer pre-computed pyramid on the local file system.

    :param str base_dir: the directory containing the pyramid
    :param bool flat: use a flat file layout (see :ref:`layouts`)
    :param bool gzip: compress chunks losslessly with gzip
    """

    can_read = True
    can_write = True

    def __init__(self, base_dir=".", flat=False, gzip=True):
        self.base_dir = base_dir
        if flat:
            self.chunk_pattern = _CHUNK_PATTERN_FLAT
        else:
            self.chunk_pattern = _CHUNK_PATTERN_SUBDIR
        self.gzip = gzip

    def fetch_info(self):
        try:
            with open(os.path.join(self.base_dir, "info")) as f:
                return json.load(f)
        except OSError as exc:
            raise DataAccessError(
                "error fetching the 'info' file from {0}" .format(
                    self.base_dir),
                exc)

    def store_info(self, info):
        try:
            os.makedirs(self.base_dir, exist_ok=True)
            with open(os.path.join(self.base_dir, "info"), "w") as f:
                json.dump(info, f, separators=(",", ":"), sort_keys=True)
        except OSError as exc:
            raise DataAccessError(
                "error storing the 'info' file in {0}" .format(
                    self.base_dir),
                exc)

    def fetch_chunk(self, key, chunk_coords):
        f = None
        for pattern in _CHUNK_PATTERN_FLAT, _CHUNK_PATTERN_SUBDIR:
            chunk_path = self._chunk_filename(key, chunk_coords, pattern)
            if os.path.isfile(chunk_path):
                f = open(chunk_path, "rb")
            elif os.path.isfile(chunk_path + ".gz"):
                f = gzip.open(chunk_path + ".gz", "rb")
        if f is None:
            raise DataAccessError("cannot find chunk {0} in {1}".format(
                self._flat_chunk_basename(key, chunk_coords), self.base_dir))
        try:
            with f:
                return f.read()
        except OSError as exc:
            raise DataAccessError(
                "error accessing chunk {0} in {1}" .format(
                    self._flat_chunk_basename(key, chunk_coords),
                    self.base_dir),
                exc)

    def store_chunk(self, buf, key, chunk_coords, already_compressed=False):
        chunk_path = self._chunk_filename(key, chunk_coords)
        try:
            os.makedirs(os.path.dirname(chunk_path), exist_ok=True)
            if self.gzip and not already_compressed:
                with gzip.open(chunk_path + ".gz", "wb") as f:
                    f.write(buf)
            else:
                with open(chunk_path, "wb") as f:
                    f.write(buf)
        except OSError as e:
            raise DataAccessError(
                "cannot store chunk {0} in {1}" .format(
                    self._flat_chunk_basename(key, chunk_coords),
                    self.base_dir),
                exc)

    def _chunk_filename(self, key, chunk_coords, pattern=None):
        if pattern is None:
            pattern = self.chunk_pattern
        xmin, xmax, ymin, ymax, zmin, zmax = chunk_coords
        chunk_filename = pattern.format(
            xmin, xmax, ymin, ymax, zmin, zmax, key=key)
        return os.path.join(self.base_dir, chunk_filename)

    def _flat_chunk_basename(self, key, chunk_coords):
        xmin, xmax, ymin, ymax, zmin, zmax = chunk_coords
        chunk_filename = _CHUNK_PATTERN_FLAT.format(
            xmin, xmax, ymin, ymax, zmin, zmax, key=key)
        return chunk_filename
