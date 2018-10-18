# Copyright (c) 2018 Forschungszentrum Juelich GmbH
# Author: Yann Leprince <y.leprince@fz-juelich.de>
#
# This software is made available under the MIT licence, see LICENCE.txt.

"""High-level access to Neuroglancer pre-computed datasets.

The central component here is the :class:`PrecomputedIO` base class. Use
:func:`get_IO_for_existing_dataset` or :func:`get_IO_for_new_dataset` for
instantiating a concrete accessor object.
"""

import json

from neuroglancer_scripts import chunk_encoding
from neuroglancer_scripts.chunk_encoding import InvalidInfoError


__all__ = [
    "get_IO_for_existing_dataset",
    "get_IO_for_new_dataset",
    "PrecomputedIO",
]


def get_IO_for_existing_dataset(accessor, encoder_options={}):
    """Create an object for accessing a pyramid with an existing *info*.

    :param Accessor accessor: a low-level accessor
    :param dict encoder_options: extrinsic encoder options
    :rtype: PrecomputedIO
    :raises DataAccessError: if the *info* file cannot be retrieved
    :raises InvalidInfoError: if the *info* file is not valid JSON
    :raises NotImplementedError: if the accessor is unable to read files
    """
    info_bytes = accessor.fetch_file("info")
    info_str = info_bytes.decode("utf-8")
    try:
        info = json.loads(info_str)
    except ValueError as exc:
        raise InvalidInfoError("Invalid JSON: {0}") from exc
    return PrecomputedIO(info, accessor, encoder_options=encoder_options)


def get_IO_for_new_dataset(info, accessor, overwrite_info=False,
                           encoder_options={}):
    """Create a new pyramid and store the provided *info*.

    :param dict info: the *info* of the new pyramid
    :param Accessor accessor: a low-level accessor
    :param dict encoder_options: extrinsic encoder options
    :raises DataAccessError: if the *info* file cannot be stored
    :raises NotImplementedError: if the accessor is unable to write files
    """
    info_str = json.dumps(info, separators=(",", ":"), sort_keys=True)
    info_bytes = info_str.encode("utf-8")
    accessor.store_file("info", info_bytes,
                        mime_type="application/json",
                        overwrite=overwrite_info)
    return PrecomputedIO(info, accessor, encoder_options=encoder_options)


class PrecomputedIO:
    """Object for high-level access to a Neuroglancer precomputed dataset.

    An object of this class provides access to chunk data in terms of NumPy
    arrays. It handles the reading/writing of files through the provided
    ``accessor``, as well as the encoding/decoding of chunks.

    The structure of the dataset (*info*) is stored in the PrecomputedIO
    instance and **must not change** during its lifetime. If you need to change
    the *info* of a dataset, use :func:`get_IO_for_new_dataset` to store the
    new *info* and create a new PrecomputedIO object.

    :param dict info: description of the dataset's structure (see :ref:`info`).

    :param Accessor accessor: an object providing low-level access to the
        dataset's files (see
        :func:`neuroglancer_scripts.accessor.get_accessor_for_url`).
    :param dict encoder_options: extrinsic encoder options (see
        :func:`neuroglancer_scripts.chunk_encoding.get_encoder`).
    """
    def __init__(self, info, accessor, encoder_options={}):
        self._info = info
        self.accessor = accessor
        self._scale_info = {
            scale_info["key"]: scale_info for scale_info in info["scales"]
        }
        self._encoders = {
            scale_info["key"]: chunk_encoding.get_encoder(info, scale_info,
                                                          encoder_options)
            for scale_info in info["scales"]
        }

    @property
    def info(self):
        """The precomputed dataset's *info* dictionary."""
        return self._info

    def scale_info(self, scale_key):
        """The *info* for a given scale.

        :param str scale_key: the *key* property of the chosen scale.
        :return: ``info["scales"][i]`` where ``info["scales"][i]["key"]
                                               == scale_key``
        :rtype: dict
        """
        return self._scale_info[scale_key]

    def scale_is_lossy(self, scale_key):
        """Test if the scale is using a lossy encoding.

        :param str scale_key: the *key* attribute of the scale
        :returns: True if the scale is using a lossy encoding
        :rtype bool:
        :raises KeyError: if the ``scale_key`` is not a valid scale of this
                          dataset
        """
        return self._encoders[scale_key].lossy

    def validate_chunk_coords(self, scale_key, chunk_coords):
        """Validate the coordinates of a chunk.

        :returns: True if the chunk coordinates are valid according to the
                  dataset's *info*
        :rtype bool:
        """
        xmin, xmax, ymin, ymax, zmin, zmax = chunk_coords
        scale_info = self.scale_info(scale_key)
        xs, ys, zs = scale_info["size"]
        if scale_info["voxel_offset"] != [0, 0, 0]:
            raise NotImplementedError("voxel_offset is not supported")
        for chunk_size in scale_info["chunk_sizes"]:
            xcs, ycs, zcs = chunk_size
            if (xmin % xcs == 0 and (xmax == min(xmin + xcs, xs))
                    and ymin % ycs == 0 and (ymax == min(ymin + ycs, ys))
                    and zmin % zcs == 0 and (zmax == min(zmin + zcs, zs))):
                return True
        return False

    def read_chunk(self, scale_key, chunk_coords):
        """Read a chunk from the dataset.

        The chunk coordinates **must** be compatible with the dataset's *info*.
        This can be checked with :meth:`validate_chunk_coords`.

        :param str scale_key: the *key* attribute of the scale
        :param tuple chunk_coords: the chunk coordinates ``(xmin, xmax, ymin,
                                   ymax, zmin, zmax)``
        :returns: chunk data contained in a 4-D NumPy array (C, Z, Y, X)
        :rtype: numpy.ndarray
        :raises DataAccessError: if the chunk's file cannot be accessed
        :raises InvalidFormatError: if the chunk cannot be decoded
        :raises AssertionError: if the chunk coordinates are incompatible with
                                the dataset's *info*
        """
        assert self.validate_chunk_coords(scale_key, chunk_coords)
        xmin, xmax, ymin, ymax, zmin, zmax = chunk_coords
        buf = self.accessor.fetch_chunk(scale_key, chunk_coords)
        encoder = self._encoders[scale_key]
        chunk = encoder.decode(buf, (xmax - xmin, ymax - ymin, zmax - zmin))
        return chunk

    def write_chunk(self, chunk, scale_key, chunk_coords):
        """Write a chunk into the dataset.

        The chunk coordinates **must** be compatible with the dataset's *info*.
        This can be checked with :meth:`validate_chunk_coords`.

        :param numpy.ndarray chunk: chunk data contained in a 4-D NumPy array
                                    (C, Z, Y, X)
        :param str scale_key: the *key* attribute of the scale
        :param tuple chunk_coords: the chunk coordinates ``(xmin, xmax, ymin,
                                   ymax, zmin, zmax)``
        :raises DataAccessError: if the chunk's file cannot be accessed
        :raises AssertionError: if the chunk coordinates are incompatible with
                                the dataset's *info*
        """
        assert self.validate_chunk_coords(scale_key, chunk_coords)
        encoder = self._encoders[scale_key]
        buf = encoder.encode(chunk)
        self.accessor.store_chunk(
            buf, scale_key, chunk_coords,
            mime_type=encoder.mime_type
        )
