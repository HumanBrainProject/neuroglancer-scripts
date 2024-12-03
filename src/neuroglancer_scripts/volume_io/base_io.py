from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class PrecompInfoScaleSharding:
    # @type cannot be easily and cleaningly serialized.
    # Pop "@type" out of dict before pass
    minishard_bits: int
    shard_bits: int
    hash: str  # use typing.Literal once 3.7 support is dropped
    minishard_index_encoding: (
        str  # use typing.Literal once 3.7 support is dropped
    )
    data_encoding: str  # use typing.Literal once 3.7 support is dropped
    preshift_bits: int


@dataclass
class PrecompInfoScale:
    chunk_sizes: List[List[int]]
    encoding: str  # use typing.Literal once 3.7 support is dropped
    key: str
    resolution: List[int]
    size: List[int]
    voxel_offset: List[float]
    sharding: Optional[PrecompInfoScaleSharding] = None


@dataclass
class PrecompInfo:
    type: str  # use typing.Literal once 3.7 support is dropped
    data_type: str  # use typing.Literal once 3.7 support is dropped
    num_channels: int
    scales: List[PrecompInfoScale]


class MultiResIOBase(ABC):

    @property
    @abstractmethod
    def info(self):
        raise NotImplementedError

    @abstractmethod
    def read_chunk(self, scale_key, chunk_coords):
        raise NotImplementedError

    @abstractmethod
    def write_chunk(self, chunk, scale_key, chunk_coords):
        raise NotImplementedError

    def iter_scale(self):
        for scale in self.info.get("scales"):
            yield scale.get("key"), scale

    def scale_info(self, scale_key):
        """The *info* for a given scale.

        :param str scale_key: the *key* property of the chosen scale.
        :return: ``info["scales"][i]`` where ``info["scales"][i]["key"]
                                               == scale_key``
        :rtype: dict
        """
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
