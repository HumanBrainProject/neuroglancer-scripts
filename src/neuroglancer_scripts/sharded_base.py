from typing import Literal, Union, List
from abc import ABC
import math
import numpy as np

_MAX_UINT64 = 0xffffffffffffffff

# spec from https://github.com/google/neuroglancer/blob/master/src/neuroglancer/datasource/precomputed/sharded.md
EncodingType = Union[Literal["raw"], Literal["gzip"]]
HashType = Union[Literal["identity"], Literal["murmurhash3_x86_128"]] 
_VALID_ENCODING = ("gzip", "raw")

class PrecomputedShardSpec:
    def __init__(self, minishard_bits: int, shard_bits: int,
        hash:HashType = "identity",
        minishard_index_encoding: EncodingType = "raw",
        data_encoding: EncodingType = "raw",
        preshift_bits: int = 0) -> None:
        
        self.minishard_bits = np.uint64(minishard_bits)
        self.shard_bits = np.uint64(shard_bits)
        self.hash = hash
        self.minishard_index_encoding = minishard_index_encoding
        self.data_encoding = data_encoding
        self.preshift_bits = preshift_bits

        self._validate()

    def _validate(self):
        assert self.minishard_bits >= 0, "minishard_bits must be >= 0"
        assert self.shard_bits >= 0, "shard_bits must be >= 0"
        assert self.hash == "identity", "Only identity hash is supported at the moment"
        assert self.preshift_bits == 0, f"non zero preshifted bits is not yet supported"
        assert self.data_encoding in _VALID_ENCODING
        assert self.minishard_index_encoding in _VALID_ENCODING
    
    def to_dict(self):
        return {
            "minishard_bits": int(self.minishard_bits),
            "shard_bits": int(self.shard_bits),
            "hash": self.hash,
            "minishard_index_encoding": self.minishard_index_encoding,
            "data_encoding": self.data_encoding,
            "preshift_bits": self.preshift_bits,
        }

    @property
    def shard_mask(self) -> np.uint64:
        if not hasattr(self, "_shard_mask"):
            movement = np.uint64(self.minishard_bits + self.shard_bits)
            mask = ~(np.uint64(_MAX_UINT64) >> movement << movement)
            self._shard_mask = mask & (~self.minishard_mask)
        return self._shard_mask
    
    @property
    def minishard_mask(self) -> np.uint64:
        if not hasattr(self, "_minishard_mask"):
            movement = np.uint64(self.minishard_bits)
            self._minishard_mask = ~(np.uint64(_MAX_UINT64) >> movement << movement)
        return self._minishard_mask

class CompressedMortonCodeBase(ABC):
    """Base abstract class for compressed morton code accessor

    :param List[int] grid_sizes: num of chunks in x,y,z. e.g. 0,64 64,128 0,64 will become 0,1,0
    """

    def __init__(self, grid_sizes: List[int]):
        self.grid_sizes = grid_sizes

        assert (
            self.grid_sizes
            and len(self.grid_sizes) == 3
            and all(isinstance(v, int)
        ) for v in self.grid_sizes), f"grid_sizes must be defined, and must be len 3 all int (should be math.ceil(sizes / chunk_sizes))"

        self.num_bits = [math.ceil(math.log2(grid_size)) for grid_size in self.grid_sizes]
        assert sum(self.num_bits) <= 64, f"Cannot use sharded file accessor for self.grid_sizes {self.grid_sizes}. It requires {self.num_bits}, larger than the max possible, 64."
    
    def compressed_morton_code(self, grid_coords: List[int]):
        assert all(grid_coord <= grid_size for grid_coord, grid_size in zip(grid_coords, self.grid_sizes)), f"{grid_coords!r} must be element-wise less or eq to {self.grid_sizes!r}, but is not"
        
        j = np.uint64(0)
        one = np.uint64(1)
        code = np.uint64(0)
        
        for i in range(max(self.num_bits)):
            for dim in range(3):
                if 2 ** i < self.grid_sizes[dim]:
                    bit = (((np.uint64(grid_coords[dim]) >> np.uint64(i)) & one) << j)
                    code |= bit
                    j += one
        return code