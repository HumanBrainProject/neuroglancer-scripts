import json
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Tuple

import numpy as np

from neuroglancer_scripts.accessor import Accessor
from neuroglancer_scripts.chunk_encoding import ChunkEncoder
from neuroglancer_scripts.volume_io.base_io import MultiResIOBase

logger = logging.getLogger(__name__)

# zarr v2 spec at https://zarr-specs.readthedocs.io/en/latest/v2/v2.0.html
# specifically, the ngff v0.4 implementation of zarr v2 https://ngff.openmicroscopy.org/0.4/


class ZarrV2IO(MultiResIOBase):

    UNIT_TO_NM = {"micrometer": 1e3}

    def __init__(self, accessor: Accessor):
        super().__init__()
        self.accessor = accessor
        self._zattrs = None
        self._zarrays = None
        self._zarray_info_dict = None
        self._decoder_dict = {}
        self._verify()

    def _verify(self):
        zgroup = json.loads(self.accessor.fetch_file(".zgroup"))
        assert (
            "zarr_format" in zgroup
        ), "Expected key 'zarr_format' to be in .zgroup, but was not: " + str(
            zgroup
        )
        assert (
            len(zgroup) == 1
        ), "Expected zarr_format to be the only key in .zgroup, but was not"
        assert zgroup["zarr_format"] == 2, (
            "Expected zarr v2, but was:" + zgroup["zarr_format"]
        )
        assert (
            "multiscales" in self.zattrs
        ), "Expected 'multiscales' in zattrs, but was not found"

    @property
    def zarray_info_dict(self):
        if self._zarray_info_dict is None:
            multiscale = self.multiscale

            path_to_datasets = []
            for dataset in multiscale.get("datasets", []):
                path = dataset.get("path")
                assert (
                    path is not None
                ), f"Expected path to exist, but it does not. {dataset}"
                path_to_datasets.append(path)

            # often network bound, use threads
            with ThreadPoolExecutor() as ex:
                zarray_bytes = list(
                    ex.map(
                        self.accessor.fetch_file,
                        [f"{p}/.zarray" for p in path_to_datasets],
                    )
                )
            self._zarray_info_dict = {
                path: json.loads(b)
                for path, b in zip(path_to_datasets, zarray_bytes)
            }

        return self._zarray_info_dict

    @property
    def zattrs(self):
        if self._zattrs is None:
            self._zattrs = json.loads(self.accessor.fetch_file(".zattrs"))
        return self._zattrs

    @property
    def multiscale(self):
        multiscales = self.zattrs.get("multiscales", [])
        assert (
            len(multiscales) == 1
        ), f"Expected one and only one multiscale, but got {len(multiscales)}"
        return multiscales[0]

    @property
    def zarrays(self):
        return list(self.zarray_info_dict.values())

    @property
    def info(self):
        data_type_set = {
            np.dtype(zarray.get("dtype")) for zarray in self.zarrays
        }
        assert (
            len(data_type_set) == 1
        ), f"Expected one and one type of datatype, but got {data_type_set}"
        data_type = list(data_type_set)[0]

        num_channels = 1
        axes = self.multiscale.get("axes", [])
        for channel_idx, axis in enumerate(axes):
            if axis.get("type") == "channel":
                num_channels = len(
                    {
                        zarray.get("shape", [])[channel_idx]
                        for zarray in self.zarrays
                    }
                )
                break

        datasets = self.multiscale.get("datasets", [])

        datasets_path_scale = []
        for dataset in datasets:
            path = dataset.get("path")
            coord_transforms = dataset.get("coordinateTransformations", [])
            assert (
                len(coord_transforms) == 1
            ), "Expected one and only one coord_transform"
            coord_transform = coord_transforms[0]

            resolutions = []
            scale = coord_transform.get("scale")
            for axis_idx, axis in enumerate(axes):
                if axis.get("type") != "space":
                    continue
                unit = axis.get("unit")
                assert (
                    unit in self.UNIT_TO_NM
                ), f"Expected {unit} to be found, but was not found"
                resolutions.append(scale[axis_idx] * self.UNIT_TO_NM[unit])

            datasets_path_scale.append((path, resolutions))

        return {
            "type": "image",
            "data_type": data_type.name,
            "num_channels": num_channels,
            "scales": [
                {
                    "chunk_sizes": [zarray.get("chunks")],
                    "encoding": "raw",
                    "key": datasets_path_scale[zarr_idx][0],
                    "resolution": datasets_path_scale[zarr_idx][1],
                    "size": zarray.get("shape"),
                    "voxel_offset": [0, 0, 0],
                }
                for zarr_idx, zarray in enumerate(self.zarrays)
            ],
        }

    def format_path(self, scale_key: str, chunk_coords: Tuple[int]):
        xmin, xmax, ymin, ymax, zmin, zmax = chunk_coords
        zarray_info = self.zarray_info_dict[scale_key]
        sep = zarray_info.get("dimension_separator", ".")
        chunks = zarray_info.get("chunks")
        assert chunks, "chunks key must be defined"
        block_sizex, block_sizey, block_sizez = chunks

        assert xmin >= 0 and ymin >= 0 and zmin >= 0, "{x,y,z}min must be >= 0"
        return (
            f"{scale_key}/{xmin // block_sizex}{sep}{ymin // block_sizey}"
            + f"{sep}{zmin // block_sizez}"
        )

    def get_encoder(self, scale_key: str):
        if scale_key in self._decoder_dict:
            return self._decoder_dict[scale_key]
        zarray_info = self.zarray_info_dict[scale_key]
        compressor = zarray_info.get("compressor")
        id = (compressor or {}).get("id")

        dtype = zarray_info.get("dtype")
        assert dtype, "dtype must be defined."
        data_type = np.dtype(dtype)

        # TODO get num of channel properly
        encoder = ChunkEncoder.get_encoder(id, data_type, 1)
        self._decoder_dict[scale_key] = encoder
        return encoder

    def read_chunk(self, scale_key: str, chunk_coords: Tuple[int]):
        path = self.format_path(scale_key, chunk_coords)
        zarray_info = self.zarray_info_dict[scale_key]
        encoder = self.get_encoder(scale_key)

        b = self.accessor.fetch_file(path)
        return encoder.decode(b, zarray_info.get("chunks"))

    def write_chunk(
        self, chunk: bytes, scale_key: str, chunk_coords: Tuple[int]
    ):
        encoder = self.get_encoder(scale_key)
        path = self.format_path(scale_key, chunk_coords)
        buf = encoder.encode(chunk)
        self.accessor.store_file(path, buf)
