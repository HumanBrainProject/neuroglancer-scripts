# Copyright (c) 2016, 2017, 2018 Forschungszentrum Juelich GmbH
# Author: Yann Leprince <y.leprince@fz-juelich.de>
#
# This software is made available under the MIT licence, see LICENCE.txt.

"""Access to a Neuroglancer pre-computed dataset on the local filesystem.

See the :mod:`~neuroglancer_scripts.accessor` module for a description of the
API.
"""

import gzip
import os.path

import neuroglancer_scripts.accessor
from neuroglancer_scripts.accessor import _CHUNK_PATTERN_FLAT, DataAccessError


__all__ = [
    "FileAccessor",
]


_CHUNK_PATTERN_SUBDIR = "{key}/{0}-{1}/{2}-{3}/{4}-{5}"

NO_COMPRESS_MIME_TYPES = {
    "application/json",
    "image/jpeg",
    "image/png",
}


class FileAccessor(neuroglancer_scripts.accessor.Accessor):
    """Access a Neuroglancer pre-computed pyramid on the local file system.

    :param str base_dir: the directory containing the pyramid
    :param bool flat: use a flat file layout (see :ref:`layouts`)
    :param bool gzip: compress chunks losslessly with gzip
    """

    can_read = True
    can_write = True

    def __init__(self, base_dir, flat=False, gzip=True):
        self.base_dir = base_dir
        if flat:
            self.chunk_pattern = _CHUNK_PATTERN_FLAT
        else:
            self.chunk_pattern = _CHUNK_PATTERN_SUBDIR
        self.gzip = gzip

    def fetch_file(self, relative_path):
        file_path = os.path.join(self.base_dir, relative_path)
        if not file_path.startswith(os.path.join(self.base_dir, "")):
            raise ValueError("only relative paths pointing under base_dir are "
                             "accepted")
        if os.path.isfile(file_path):
            f = open(file_path, "rb")
        elif os.path.isfile(file_path + ".gz"):
            f = gzip.open(file_path + ".gz", "rb")
        else:
            raise DataAccessError("Cannot find {0} in {1}".format(
                relative_path, self.base_dir))
        try:
            with f:
                return f.read()
        except OSError as exc:
            raise DataAccessError(
                "Error fetching {1}: {2}" .format(
                    relative_path, self.base_dir, exc)) from exc

    def store_file(self, relative_path, buf,
                   mime_type="application/octet-stream",
                   overwrite=False):
        file_path = os.path.join(self.base_dir, relative_path)
        if not file_path.startswith(self.base_dir + os.path.sep):
            raise ValueError("only relative paths pointing under base_dir are "
                             "accepted")
        mode = "wb" if overwrite else "xb"
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            if self.gzip and mime_type not in NO_COMPRESS_MIME_TYPES:
                with gzip.open(file_path + ".gz", mode) as f:
                    f.write(buf)
            else:
                with open(file_path, mode) as f:
                    f.write(buf)
        except OSError as exc:
            raise DataAccessError("Error storing {0}: {1}"
                                  .format(file_path, exc)) from exc

    def fetch_chunk(self, key, chunk_coords):
        f = None
        for pattern in _CHUNK_PATTERN_FLAT, _CHUNK_PATTERN_SUBDIR:
            chunk_path = self._chunk_filename(key, chunk_coords, pattern)
            if os.path.isfile(chunk_path):
                f = open(chunk_path, "rb")
            elif os.path.isfile(chunk_path + ".gz"):
                f = gzip.open(chunk_path + ".gz", "rb")
        if f is None:
            raise DataAccessError("Cannot find chunk {0} in {1}".format(
                self._flat_chunk_basename(key, chunk_coords), self.base_dir))
        try:
            with f:
                return f.read()
        except OSError as exc:
            raise DataAccessError(
                "Error accessing chunk {0} in {1}: {2}" .format(
                    self._flat_chunk_basename(key, chunk_coords),
                    self.base_dir, exc)) from exc

    def store_chunk(self, buf, key, chunk_coords,
                    mime_type="application/octet-stream",
                    overwrite=True):
        chunk_path = self._chunk_filename(key, chunk_coords)
        mode = "wb" if overwrite else "xb"
        try:
            os.makedirs(os.path.dirname(chunk_path), exist_ok=True)
            if self.gzip and mime_type not in NO_COMPRESS_MIME_TYPES:
                with gzip.open(chunk_path + ".gz", mode) as f:
                    f.write(buf)
            else:
                with open(chunk_path, mode) as f:
                    f.write(buf)
        except OSError as exc:
            raise DataAccessError(
                "Error storing chunk {0} in {1}: {2}" .format(
                    self._flat_chunk_basename(key, chunk_coords),
                    self.base_dir, exc)) from exc

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
