import json
import struct
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import List

from neuroglancer_scripts.accessor import Accessor
from neuroglancer_scripts.chunk_encoding import (
    ChunkEncoder,
)
from neuroglancer_scripts.volume_io.base_io import MultiResIOBase

# N5 spec: https://github.com/saalfeldlab/n5
# supporting an (undocumented?) custom group per
# https://github.com/bigdataviewer/bigdataviewer-core/blob/master/BDV%20N5%20format.md
# https://github.com/saalfeldlab/n5-viewer
#


@dataclass
class N5ScaleAttr:
    pass


@dataclass
class N5RootAttr:
    downsamplingFactors: List[List[int]]  # noqa: N815
    # once py3.7 is dropped, use Literal instead
    # {uint8, uint16, uint32, uint64, int8, int16, int32, int64, float32,}
    # {float64}
    dataType: str  # noqa: N815
    multiScale: bool  # noqa: N815
    resolution: List[int]
    unit: List[str]  # seems to be... neuroglancer specific?


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

        encoder = ChunkEncoder.get_encoder(compression_type, datatype, 1)

        self._decoder_dict[scale_key] = encoder

        return encoder

    @property
    def attribute_json(self):
        if self._attributes_json is None:
            self._attributes_json = json.loads(
                self.accessor.fetch_file("attributes.json")
            )
        return self._attributes_json

    @property
    def info(self):
        downsample_factors = self.attribute_json.get("downsamplingFactors", [])
        resolution = self.attribute_json.get("resolution", [])
        unit = self.attribute_json.get("unit", [])

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
                        * self.UNIT_TO_NM.get(unit[order_idx], 1)
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
