import json
import struct
from concurrent.futures import ThreadPoolExecutor

from neuroglancer_scripts.accessor import Accessor
from neuroglancer_scripts.chunk_encoding import (
    BloscEncoder,
    GZipEncoder,
    RawChunkEncoder,
)
from neuroglancer_scripts.iobase import MultiResIOBase

# N5 spec: https://github.com/saalfeldlab/n5


class N5IO(MultiResIOBase):

    UNIT_TO_NM = {"um": 1e3}

    def __init__(
        self, attributes_json, accessor: Accessor, encoder_options={}
    ):
        super().__init__()
        self._attributes_json = attributes_json
        self.accessor = accessor
        assert accessor.can_read, "N5IO must have readable accessor"

        # networkbound, use threads
        with ThreadPoolExecutor() as ex:
            self.scale_attributes = [
                json.loads(attr)
                for attr in list(
                    ex.map(
                        accessor.fetch_file,
                        [
                            f"s{str(idx)}/attributes.json"
                            for idx, _ in enumerate(
                                self.attribute_json.get(
                                    "downsamplingFactors", []
                                )
                            )
                        ],
                    )
                )
            ]

        self._scale_attributes_dict = {
            f"s{idx}": scale_attribute
            for idx, scale_attribute in enumerate(self.scale_attributes)
        }

        self._decoder_dict = {}

    def _get_encoder(self, scale_key: str):

        if scale_key in self._decoder_dict:
            return self._decoder_dict[scale_key]

        compression_type = self._scale_attributes_dict[scale_key][
            "compression"
        ]["type"]
        datatype = self._scale_attributes_dict[scale_key]["dataType"]

        encoder = None
        if compression_type == "blosc":
            encoder = BloscEncoder(datatype, 1)
        if compression_type == "gzip":
            encoder = GZipEncoder(datatype, 1)
        if compression_type == "raw":
            encoder = RawChunkEncoder(datatype, 1)

        if encoder is None:
            raise NotImplementedError(f"Cannot parse {compression_type}")

        self._decoder_dict[scale_key] = encoder
        return encoder

    @property
    def attribute_json(self):
        return self._attributes_json

    def iter_scale(self):
        for scale in self.info.get("scales"):
            yield scale.get("key"), scale

    @property
    def info(self):
        downsample_factors = self.attribute_json.get("downsamplingFactors", [])
        resolution = self.attribute_json.get("resolution", [])
        units = self.attribute_json.get("units", [])

        return {
            "type": "image",
            "data_type": self.scale_attributes[0].get("dataType"),
            "num_channels": 1,
            "scales": [
                {
                    "chunk_sizes": [scale_attribute.get("blockSize")],
                    "encoding": scale_attribute.get("compression", {}).get(
                        "type"
                    ),
                    "key": f"s{str(scale_idx)}",
                    "resolution": [
                        res
                        * downsample_factors[scale_idx][order_idx]
                        * self.UNIT_TO_NM.get(units[order_idx], 1)
                        for order_idx, res in enumerate(resolution)
                    ],
                    "size": scale_attribute.get("dimensions"),
                    "voxel_offset": [0, 0, 0],
                }
                for scale_idx, scale_attribute in enumerate(
                    self.scale_attributes
                )
            ],
        }

    def scale_info(self, scale_key):
        found_scale = [
            scale
            for scale in self.info.get("scales", [])
            if scale.get("key") == scale_key
        ]
        if len(found_scale) == 0:
            raise IndexError(f"Cannot find {scale_key}")
        if len(found_scale) > 1:
            raise IndexError(f"Found multiple {scale_key}")
        return found_scale[0]

    def _get_grididx_from_chunkcoord(self, scale_key, chunk_coords):
        xmin, xmax, ymin, ymax, zmin, zmax = chunk_coords
        block_sizex, block_sizey, block_sizez = self._scale_attributes_dict[
            scale_key
        ].get("blockSize")
        return xmin // block_sizex, ymin // block_sizey, zmin // block_sizez

    def read_chunk(self, scale_key, chunk_coords):
        gridx, gridy, gridz = self._get_grididx_from_chunkcoord(
            scale_key, chunk_coords
        )
        chunk = self.accessor.fetch_file(
            f"{scale_key}/{gridx}/{gridy}/{gridz}"
        )
        mode, dim, sizex, sizey, sizez = struct.unpack(">HHIII", chunk[:16])
        assert (
            dim == 3
        ), "N5 currently can only handle single channel, three dimension array"
        encoder = self._get_encoder(scale_key)
        return encoder.decode(chunk[16:], (sizex, sizey, sizez))

    def write_chunk(self, chunk, scale_key, chunk_coords):
        assert (
            self.accessor.can_write
        ), "N5IO.write_chunk: accessor cannot write"
        xmin, xmax, ymin, ymax, zmin, zmax = chunk_coords
        gridx, gridy, gridz = self._get_grididx_from_chunkcoord(
            scale_key, chunk_coords
        )
        encoder = self._get_encoder(scale_key)
        buf = encoder.encode(chunk)
        hdr = struct.pack(
            ">HHIII", (0, 3, xmax - xmin, ymax - ymin, zmax - zmin)
        )
        self.accessor.store_file(
            f"{scale_key}/{gridx}/{gridy}/{gridz}", hdr + buf
        )
