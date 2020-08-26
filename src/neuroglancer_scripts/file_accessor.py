# Copyright (c) 2016, 2017, 2018 Forschungszentrum Juelich GmbH
# Author: Yann Leprince <y.leprince@fz-juelich.de>
#
# This software is made available under the MIT licence, see LICENCE.txt.

"""Access to a Neuroglancer pre-computed dataset on the local filesystem.

See the :mod:`~neuroglancer_scripts.accessor` module for a description of the
API.
"""

import gzip
import os
import pathlib

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

    :param str base_dir: path to the directory containing the pyramid
    :param bool flat: use a flat file layout (see :ref:`layouts`)
    :param bool gzip: compress chunks losslessly with gzip
    """

    can_read = True
    can_write = True

    def __init__(self, base_dir, flat=False, gzip=True):
        self.base_path = pathlib.Path(base_dir)
        if flat:
            self.chunk_pattern = _CHUNK_PATTERN_FLAT
        else:
            self.chunk_pattern = _CHUNK_PATTERN_SUBDIR
        self.gzip = gzip

    def file_exists(self, relative_path):
        relative_path = pathlib.Path(relative_path)
        file_path = self.base_path / relative_path
        if ".." in file_path.relative_to(self.base_path).parts:
            raise ValueError("only relative paths pointing under base_path "
                             "are accepted")
        try:
            if file_path.is_file():
                return True
            elif file_path.with_name(file_path.name + ".gz").is_file():
                return True
        except OSError as exc:
            raise DataAccessError(
                "Error fetching {0}: {1}".format(file_path, exc)) from exc
        return False

    def fetch_file(self, relative_path):
        relative_path = pathlib.Path(relative_path)
        file_path = self.base_path / relative_path
        if ".." in file_path.relative_to(self.base_path).parts:
            raise ValueError("only relative paths pointing under base_path "
                             "are accepted")
        try:
            if file_path.is_file():
                f = file_path.open("rb")
            elif file_path.with_name(file_path.name + ".gz").is_file():
                f = gzip.open(str(file_path.with_name(file_path.name + ".gz")),
                              "rb")
            else:
                raise DataAccessError("Cannot find {0} in {1}".format(
                    relative_path, self.base_path))
            with f:
                return f.read()
        except OSError as exc:
            raise DataAccessError(
                "Error fetching {0}: {1}".format(file_path, exc)) from exc

    def store_file(self, relative_path, buf,
                   mime_type="application/octet-stream",
                   overwrite=False):
        relative_path = pathlib.Path(relative_path)
        file_path = self.base_path / relative_path
        if ".." in file_path.relative_to(self.base_path).parts:
            raise ValueError("only relative paths pointing under base_path "
                             "are accepted")
        mode = "wb" if overwrite else "xb"
        try:
            os.makedirs(str(file_path.parent), exist_ok=True)
            if self.gzip and mime_type not in NO_COMPRESS_MIME_TYPES:
                with gzip.open(
                        str(file_path.with_name(file_path.name + ".gz")),
                        mode) as f:
                    f.write(buf)
            else:
                with file_path.open(mode) as f:
                    f.write(buf)
        except OSError as exc:
            raise DataAccessError("Error storing {0}: {1}"
                                  .format(file_path, exc)) from exc

    def fetch_chunk(self, key, chunk_coords):
        f = None
        try:
            for pattern in _CHUNK_PATTERN_FLAT, _CHUNK_PATTERN_SUBDIR:
                chunk_path = self._chunk_path(key, chunk_coords, pattern)
                if chunk_path.is_file():
                    f = chunk_path.open("rb")
                elif chunk_path.with_name(chunk_path.name + ".gz").is_file():
                    f = gzip.open(
                        str(chunk_path.with_name(chunk_path.name + ".gz")),
                        "rb"
                    )
            if f is None:
                raise DataAccessError(
                    "Cannot find chunk {0} in {1}" .format(
                        self._flat_chunk_basename(key, chunk_coords),
                        self.base_path)
                )
            with f:
                return f.read()
        except OSError as exc:
            raise DataAccessError(
                "Error accessing chunk {0} in {1}: {2}" .format(
                    self._flat_chunk_basename(key, chunk_coords),
                    self.base_path, exc)) from exc

    def store_chunk(self, buf, key, chunk_coords,
                    mime_type="application/octet-stream",
                    overwrite=True):
        chunk_path = self._chunk_path(key, chunk_coords)
        mode = "wb" if overwrite else "xb"
        try:
            os.makedirs(str(chunk_path.parent), exist_ok=True)
            if self.gzip and mime_type not in NO_COMPRESS_MIME_TYPES:
                with gzip.open(
                        str(chunk_path.with_name(chunk_path.name + ".gz")),
                        mode) as f:
                    f.write(buf)
            else:
                with chunk_path.open(mode) as f:
                    f.write(buf)
        except OSError as exc:
            raise DataAccessError(
                "Error storing chunk {0} in {1}: {2}" .format(
                    self._flat_chunk_basename(key, chunk_coords),
                    self.base_path, exc)) from exc

    def _chunk_path(self, key, chunk_coords, pattern=None):
        if pattern is None:
            pattern = self.chunk_pattern
        xmin, xmax, ymin, ymax, zmin, zmax = chunk_coords
        chunk_filename = pattern.format(
            xmin, xmax, ymin, ymax, zmin, zmax, key=key)
        return self.base_path / chunk_filename

    def _flat_chunk_basename(self, key, chunk_coords):
        xmin, xmax, ymin, ymax, zmin, zmax = chunk_coords
        chunk_filename = _CHUNK_PATTERN_FLAT.format(
            xmin, xmax, ymin, ymax, zmin, zmax, key=key)
        return chunk_filename
