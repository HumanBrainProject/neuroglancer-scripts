# Copyright (c) 2018 Forschungszentrum Juelich GmbH
# Author: Yann Leprince <y.leprince@fz-juelich.de>
#
# This software is made available under the MIT licence, see LICENCE.txt.

import numpy as np


__all__ = [
    "get_encoder",
    "add_argparse_options",
    "IncompatibleEncoderError",
    "InvalidInfoError",
    "RawChunkEncoder",
    "CompressedSegmentationEncoder",
    "JpegChunkEncoder",
]


# TODO move to a data_type module
"""List of possible values for ``data_type``."""
NEUROGLANCER_DATA_TYPES = ("uint8", "uint16", "uint32", "uint64", "float32")


def get_encoder(info, scale_info, encoder_options={}):
    """Create an Encoder object for the provided scale.

    :param dict info: a Neuroglancer *info* dictionary (:ref:`info`) containing
                      general encoding parameters (:attr:`data_type` and
                      :attr:`num_channels`)
    :param dict scale_info: an element of (``info["scales"]``) containing
                            scale-specific encoding parameters
                            (:attr:`encoding` and encoding-specific parameters)
    :param dict encoder_params: extrinsic encoder parameters
    """
    data_type = info["data_type"]
    num_channels = info["num_channels"]
    encoding = scale_info["encoding"]
    if not isinstance(num_channels, int) or not num_channels > 0:
        raise InvalidInfoError("invalid value {0} for num_channels (must be "
                               "a positive integer)".format(num_channels))
    if data_type not in NEUROGLANCER_DATA_TYPES:
        raise InvalidInfoError("invalid data_type {0} (should be one of {1})"
                               .format(data_type, NEUROGLANCER_DATA_TYPES))
    if encoding == "raw":
        return RawChunkEncoder(data_type, num_channels)
    elif encoding == "compressed_segmentation":
        try:
            block_size = scale_info["compressed_segmentation_block_size"]
        except KeyError:
            raise InvalidInfoError(
                'encoding is set to "compressed_segmentation" but '
                '"compressed_segmentation_block_size" is missing')
        return CompressedSegmentationEncoder(data_type, num_channels,
                                             block_size)
    elif encoding == "jpeg":
        jpeg_plane = encoder_options.get("jpeg_plane", "xy")
        jpeg_quality = encoder_options.get("jpeg_quality", 95)
        return JpegChunkEncoder(data_type, num_channels,
                                jpeg_plane=jpeg_plane,
                                jpeg_quality=jpeg_quality)
    else:
        raise InvalidInfoError("Invalid encoding {0}".format(encoding))


def add_argparse_options(parser, allow_lossy):
    if allow_lossy:
        group = parser.add_argument_group("Options for JPEG compression")
        group.add_argument("--jpeg-quality", type=int, default=95, metavar="Q",
                           help="JPEG quality factor (from 0 to 100, values "
                           "above 95 provide little extra quality but "
                           "increase file size)")
        group.add_argument("--jpeg-plane", choices=("xy", "xz"), default="xy",
                           help='plane of JPEG compression (default: xy)')


class IncompatibleEncoderError(Exception):
    pass


class InvalidInfoError(Exception):
    pass


class ChunkEncoder:
    def __init__(self, data_type, num_channels):
        assert num_channels > 0
        self.num_channels = num_channels
        self.dtype = np.dtype(data_type).newbyteorder("<")


class RawChunkEncoder(ChunkEncoder):
    lossy = False
    already_compressed = False

    def encode(self, chunk):
        assert np.can_cast(chunk.dtype, self.dtype, casting="safe")
        assert chunk.ndim == 4
        assert chunk.shape[0] == self.num_channels
        buf = chunk.tobytes()
        return buf

    def decode(self, buf, chunk_size):
        return np.frombuffer(buf, dtype=self.dtype).reshape(
            (self.num_channels, chunk_size[2], chunk_size[1], chunk_size[0]))


class CompressedSegmentationEncoder(ChunkEncoder):
    lossy = False
    already_compressed = False

    def __init__(self, data_type, num_channels, block_size):
        if data_type not in ("uint32", "uint64"):
            raise IncompatibleEncoderError(
                "compressed_segmentation encoding can only handle uint32 or "
                "uint64 data_type")
        super().__init__(data_type, num_channels)
        self.block_size = block_size

    def encode(self, chunk):
        from neuroglancer_scripts import _compressed_segmentation
        assert np.can_cast(chunk.dtype, self.dtype, casting="safe")
        assert chunk.ndim == 4
        assert chunk.shape[0] == self.num_channels
        buf = _compressed_segmentation.encode_chunk(chunk, self.block_size)
        return buf

    def decode(self, buf, chunk_size):
        from neuroglancer_scripts import _compressed_segmentation
        chunk = np.empty(
            (self.num_channels, chunk_size[2], chunk_size[1], chunk_size[0]),
            dtype=self.dtype
        )
        _compressed_segmentation.decode_chunk_into(chunk, buf, self.block_size)
        return chunk


class JpegChunkEncoder(ChunkEncoder):
    lossy = False
    already_compressed = True

    def __init__(self, data_type, num_channels, jpeg_quality, jpeg_plane):
        if data_type != "uint8" or num_channels not in (1, 3):
            raise IncompatibleEncoderError(
                "JPEG encoding can only handle uint8 data_type with 1 or 3 "
                "channels")
        super().__init__(data_type, num_channels)
        self.jpeg_quality = jpeg_quality
        self.jpeg_plane = jpeg_plane

    def encode(self, chunk):
        from neuroglancer_scripts import _jpeg
        assert np.can_cast(chunk.dtype, self.dtype, casting="safe")
        assert chunk.ndim == 4
        assert chunk.shape[0] == self.num_channels
        buf = _jpeg.encode_chunk(chunk, self.jpeg_quality, self.jpeg_plane)
        return buf

    def decode(self, buf, chunk_size):
        from neuroglancer_scripts import _jpeg
        return _jpeg.decode_chunk(buf, chunk_size, self.num_channels)
