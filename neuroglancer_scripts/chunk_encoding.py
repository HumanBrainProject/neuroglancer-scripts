# Copyright (c) 2018 Forschungszentrum Juelich GmbH
# Author: Yann Leprince <y.leprince@fz-juelich.de>
#
# This software is made available under the MIT licence, see LICENCE.txt.

import numpy as np


def get_encoder(info, scale_info, encoder_params={}):
    data_type = info["data_type"]
    num_channels = info["num_channels"]
    encoding = scale_info["encoding"]
    if encoding == "raw":
        return RawChunkEncoder(data_type, num_channels)
    elif encoding == "compressed_segmentation":
        block_size = scale_info["compressed_segmentation_block_size"]
        return CompressedSegmentationEncoder(data_type, num_channels,
                                             block_size)
    elif encoding == "jpeg":
        # TODO properly handle missing params
        jpeg_quality = encoder_params["jpeg_quality"]
        jpeg_plane = encoder_params["jpeg_plane"]
        return JpegChunkEncoder(data_type, num_channels,
                                jpeg_quality, jpeg_plane)
    else:
        return RuntimeError("Invalid encoding")  # TODO appropriate error type?


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


class ChunkEncoder:
    def __init__(self, data_type, num_channels):
        assert num_channels > 0
        self.num_channels = num_channels
        self.dtype = np.dtype(data_type).newbyteorder("<")


class RawChunkEncoder(ChunkEncoder):
    lossy = False
    already_compressed = False

    def encode(self, chunk):
        assert chunk.dtype == self.dtype
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
        from . import compressed_segmentation
        assert chunk.dtype == self.dtype
        assert chunk.ndim == 4
        assert chunk.shape[0] == self.num_channels
        buf = compressed_segmentation.encode_chunk(chunk, self.block_size)
        return buf

    def decode(self, buf, chunk_size):
        from . import compressed_segmentation
        chunk = np.empty(
            (self.num_channels, chunk_size[2], chunk_size[1], chunk_size[0]),
            dtype=self.dtype
        )
        compressed_segmentation.decode_chunk_into(chunk, buf, self.block_size)
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
        from . import jpeg
        assert chunk.dtype == self.dtype
        assert chunk.ndim == 4
        assert chunk.shape[0] == self.num_channels
        buf = jpeg.encode_chunk(chunk, self.jpeg_quality, self.jpeg_plane)
        self.accessor.store_chunk(buf, key, xmin, xmax, ymin, ymax, zmin, zmax)

    def decode(self, buf, chunk_size):
        raise NotImplementedError("Decoding JPEG chunks is unsupported, "
                                  "please process your data in raw format and "
                                  "convert to JPEG as a last step")
