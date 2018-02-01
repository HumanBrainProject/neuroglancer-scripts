# Copyright (c) 2018 Forschungszentrum Juelich GmbH
# Author: Yann Leprince <y.leprince@fz-juelich.de>
#
# This software is made available under the MIT licence, see LICENCE.txt.

from . import chunk_encoding

# TODO rename
# TODO decide what goes in this class: chunk layout + fetching into buffers (+ iteration?) vs chunk encoding/decoding
class ChunkIo:
    def __init__(self, info, accessor, encoder_params={}):
        self.accessor = accessor
        self.encoders = {
            scale_info["key"]: chunk_encoding.get_encoder(
                info, scale_info, encoder_params)
            for scale_info in info["scales"]
        }

    def read_chunk(self, key, chunk_coords):
        xmin, xmax, ymin, ymax, zmin, zmax = chunk_coords
        buf = self.accessor.fetch_chunk(key, chunk_coords)
        encoder = self.encoders[key]
        chunk = encoder.decode(buf, (xmax - xmin, ymax - ymin, zmax - zmin))
        return chunk

    def write_chunk(self, chunk, key, chunk_coords):
        encoder = self.encoders[key]
        buf = encoder.encode(chunk)
        self.accessor.store_chunk(
            buf, key, chunk_coords,
            already_compressed=encoder.already_compressed
        )
