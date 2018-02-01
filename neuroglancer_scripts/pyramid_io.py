# Copyright (c) 2018 Forschungszentrum Juelich GmbH
# Author: Yann Leprince <y.leprince@fz-juelich.de>
#
# This software is made available under the MIT licence, see LICENCE.txt.

from . import chunk_encoding


# TODO decide what goes in this class: chunk layout + fetching into buffers +
# iteration?
class PrecomputedPyramidIo:
    def __init__(self, info, accessor, encoder_params={}):
        self._info = info
        self.accessor = accessor
        self.scale_info = {
            scale_info["key"]: scale_info for scale_info in info["scales"]
        }
        self.encoders = {
            scale_info["key"]: chunk_encoding.get_encoder(info, scale_info,
                                                          encoder_params)
            for scale_info in info["scales"]
        }

    @property
    def info(self):
        return self._info

    def scale_info(self, key):
        return self.scale_info[key]

    def validate_chunk_coords(self, key, chunk_coords):
        xmin, xmax, ymin, ymax, zmin, zmax = chunk_coords
        scale_info = self.scale_info[key]
        xs, ys, zs = scale_info["size"]
        if scale_info["voxel_offset"] != [0, 0, 0]:
            raise NotImplementedError("voxel_offset is not supported")
        for chunk_size in scale_info["chunk_sizes"]:
            xcs, ycs, zcs = chunk_size
            if (xmin // xs == 0 and (((xmax + 1) // xs == 0) or xmax == xs)
                and ymin // ys == 0 and (((ymax + 1) // ys == 0) or ymax == ys)
                and zmin // zs == 0 and (((zmax + 1) // zs == 0) or zmax == zs)):
                return True
        return False

    def read_chunk(self, key, chunk_coords):
        assert self.validate_chunk_coords(key, chunk_coords)
        xmin, xmax, ymin, ymax, zmin, zmax = chunk_coords
        buf = self.accessor.fetch_chunk(key, chunk_coords)
        encoder = self.encoders[key]
        chunk = encoder.decode(buf, (xmax - xmin, ymax - ymin, zmax - zmin))
        return chunk

    def write_chunk(self, chunk, key, chunk_coords):
        assert self.validate_chunk_coords(key, chunk_coords)
        encoder = self.encoders[key]
        buf = encoder.encode(chunk)
        self.accessor.store_chunk(
            buf, key, chunk_coords,
            already_compressed=encoder.already_compressed
        )
