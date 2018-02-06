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
    "InvalidFormatError",
    "ChunkEncoder",
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
                      general encoding parameters (``data_type`` and
                      ``num_channels``)
    :param dict scale_info: an element of (``info["scales"]``) containing
                            scale-specific encoding parameters
                            (``encoding`` and encoding-specific parameters)
    :param dict encoder_params: extrinsic encoder parameters
    :returns: an instance of a chunk encoder
    :rtype: ChunkEncoder
    :raises InvalidInfoError: if the provided *info* dict is invalid
    """
    try:
        data_type = info["data_type"]
        num_channels = info["num_channels"]
        encoding = scale_info["encoding"]
    except KeyError as exc:
        raise InvalidInfoError("The info dict is missing an essential key {0}"
                               .format(exc)) from exc
    if not isinstance(num_channels, int) or not num_channels > 0:
        raise InvalidInfoError("Invalid value {0} for num_channels (must be "
                               "a positive integer)".format(num_channels))
    if data_type not in NEUROGLANCER_DATA_TYPES:
        raise InvalidInfoError("Invalid data_type {0} (should be one of {1})"
                               .format(data_type, NEUROGLANCER_DATA_TYPES))
    try:
        if encoding == "raw":
            return RawChunkEncoder(data_type, num_channels)
        elif encoding == "compressed_segmentation":
            try:
                block_size = scale_info["compressed_segmentation_block_size"]
            except KeyError:
                raise InvalidInfoError(
                    'Encoding is set to "compressed_segmentation" but '
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
    except IncompatibleEncoderError as exc:
        raise InvalidInfoError(str(exc)) from exc


def add_argparse_options(parser, allow_lossy=False):
    """Add command-line options for chunk encoding.

    :param parser: an instance of :class:`argparse.ArgumentParser`
    :param bool allow_lossy: show parameters for lossy encodings (i.e. jpeg)

    The extrinsic encoder parameters can be obtained from command-line
    arguments with :func:`add_argparse_options` and passed to
    :func:`get_encoder`::

        import argparse
        parser = argparse.ArgumentParser()
        add_argparse_options(parser)
        args = parser.parse_args()
        get_encoder(info, scale_info, vars(args))
    """
    import argparse
    if allow_lossy:
        def jpeg_quality(arg):
            q = int(arg)
            if not 1 <= q <= 100:
                raise argparse.ArgumentTypeError(
                    "JPEG quality must be between 1 and 100")
            return q

        group = parser.add_argument_group("Options for JPEG compression")
        group.add_argument("--jpeg-quality", type=jpeg_quality,
                           default=95, metavar="Q",
                           help="JPEG quality factor (1 is worst, 100 is "
                           "best, values above 95 increase file size but "
                           "provide hardly any extra quality)")
        group.add_argument("--jpeg-plane", choices=("xy", "xz"), default="xy",
                           help='plane of JPEG compression (default: xy)')


class IncompatibleEncoderError(Exception):
    """Raised when an Encoder cannot handle the requested data type."""
    pass


class InvalidInfoError(Exception):
    """Raised when an *info* dict is invalid or inconsistent."""
    pass


class InvalidFormatError(Exception):
    """Raised when chunk data cannot be decoded properly."""
    pass


class ChunkEncoder:
    """Encode/decode chunks from NumPy arrays to byte buffers.

    :param str data_type: data type supported by Neuroglancer
    :param int num_channels: number of image channels
    """

    lossy = False
    """True if this encoder is lossy."""

    already_compressed = False
    """True if additional compression (e.g. gzip) is superfluous."""

    def __init__(self, data_type, num_channels):
        self.num_channels = num_channels
        self.dtype = np.dtype(data_type).newbyteorder("<")

    def encode(self, chunk):
        """Encode a chunk from a NumPy array into bytes.

        :param numpy.ndarray chunk: array with four dimensions (C, Z, Y, X)
        :returns: encoded chunk
        :rtype: bytes
        """
        raise NotImplementedError

    def decode(self, buf, chunk_size):
        """Decode a chunk from bytes into a NumPy array.

        :param bytes buf: encoded chunk
        :param tuple chunk_size: the 3-D size of the chunk (X, Y, Z)
        :returns: chunk contained in a 4-D NumPy array (C, Z, Y, X)
        :rtype: numpy.ndarray
        :raises InvalidFormatError: if there the chunk data cannot be decoded
                                    properly
        """
        raise NotImplementedError


class RawChunkEncoder(ChunkEncoder):
    """Codec for to the Neuroglancer raw chunk format.

    :param str data_type: data type supported by Neuroglancer
    :param int num_channels: number of image channels
    """
    lossy = False
    already_compressed = False

    def encode(self, chunk):
        chunk = np.asarray(chunk).astype(self.dtype, casting="safe")
        assert chunk.ndim == 4
        assert chunk.shape[0] == self.num_channels
        buf = chunk.tobytes()
        return buf

    def decode(self, buf, chunk_size):
        try:
            return np.frombuffer(buf, dtype=self.dtype).reshape(
                (self.num_channels,
                 chunk_size[2], chunk_size[1], chunk_size[0]))
        except Exception as exc:
            raise InvalidFormatError("Cannot decode raw-encoded chunk: {0}"
                                     .format(exc)) from exc


class CompressedSegmentationEncoder(ChunkEncoder):
    """Codec for to the Neuroglancer precomputed chunk format.

    :param str data_type: data type supported by Neuroglancer
    :param int num_channels: number of image channels
    :param list block_size: ``block_size`` for the compressed segmentation
                            compression algorithm
    :raises IncompatibleEncoderError: if data_type or num_channels are
                                      unsupported
    """
    lossy = False
    already_compressed = False

    def __init__(self, data_type, num_channels, block_size):
        if data_type not in ("uint32", "uint64"):
            raise IncompatibleEncoderError(
                "The compressed_segmentation encoding can only handle uint32 "
                "or uint64 data_type")
        super().__init__(data_type, num_channels)
        self.block_size = block_size

    def encode(self, chunk):
        from neuroglancer_scripts import _compressed_segmentation
        chunk = np.asarray(chunk).astype(self.dtype, casting="safe")
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
    """Codec for to the Neuroglancer raw chunk format.

    :param str data_type: data type supported by Neuroglancer
    :param int num_channels: number of image channels
    :param int jpeg_quality: quality factor for JPEG compression
    :param str jpeg_plane: plane of JPEG compression (``"xy"`` or ``"xz"``)
    :raises IncompatibleEncoderError: if data_type or num_channels are
                                      unsupported
    """
    lossy = False
    already_compressed = True

    def __init__(self, data_type, num_channels,
                 jpeg_quality=95, jpeg_plane="xy"):
        if data_type != "uint8" or num_channels not in (1, 3):
            raise IncompatibleEncoderError(
                "The JPEG encoding can only handle uint8 data_type with 1 or "
                "3 channels")
        super().__init__(data_type, num_channels)
        assert 1 <= jpeg_quality <= 100
        assert jpeg_plane in ("xy", "xz")
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
