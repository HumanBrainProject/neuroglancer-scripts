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

        self._shard_mask = None
        self._minishard_mask = None

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
        if not self._shard_mask:
            movement = np.uint64(self.minishard_bits + self.shard_bits)
            mask = ~(np.uint64(_MAX_UINT64) >> movement << movement)
            self._shard_mask = mask & (~self.minishard_mask)
        return self._shard_mask
    
    @property
    def minishard_mask(self) -> np.uint64:
        if not self._minishard_mask:
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

class CMCReadWrite(ABC):
    
    can_read_cmc = False
    can_write_cmc = False

    def __init__(self, shard_spec: PrecomputedShardSpec) -> None:
        self.shard_spec = shard_spec
        self.header_byte_length = int(2 ** self.shard_spec.minishard_bits * 16)

    def fetch_cmc_chunk(self, cmc: np.uint64):
        raise NotImplementedError

    def store_cmc_chunk(self, buf:bytes, cmc: np.uint64):
        raise NotImplementedError

    def close(self):
        pass

    def get_shard_key(self, cmc: np.uint64) -> np.uint64:
        return (self.shard_spec.shard_mask & cmc) >> self.shard_spec.minishard_bits

    def get_minishard_key(self, cmc: np.uint64) -> np.uint64:
        return (self.shard_spec.minishard_mask & cmc)
    
    def read_bytes(self, offset:int, length: int) -> bytes:
        raise NotImplementedError
    
    def get_minishards_offsets(self):
        assert self.can_read_cmc
        header_byte_length = self.read_bytes(0, self.header_byte_length)
        return np.frombuffer(header_byte_length, dtype=np.uint64)

class ShardedScaleBase(CompressedMortonCodeBase, CMCReadWrite):
    
    def __init__(self, key, chunk_sizes,
                 shard_spec: PrecomputedShardSpec,
                 mip_sizes=None):
        assert (
            chunk_sizes
            and len(chunk_sizes) == 3
            and all(isinstance(v, int) for v in chunk_sizes)
            and len({cz for cz in chunk_sizes}) == 1
        ), f"chunk_sizes must be defined, len 3 and all int. chunk_sizes must be same in all dimensions.: {chunk_sizes}"
        self.chunk_sizes = chunk_sizes

        grid_sizes = [math.ceil(mipsize / chunksize) for mipsize, chunksize in zip(mip_sizes, chunk_sizes)]
        CompressedMortonCodeBase.__init__(self, grid_sizes)
        CMCReadWrite.__init__(self, shard_spec)

        self.key = key
    
    def get_cmc(self, chunk_coords) -> np.uint64:
        
        xmin, xmax, ymin, ymax, zmin, zmax = chunk_coords
        xcs, ycs, zcs = self.chunk_sizes
        
        assert xmin % xcs == 0, f"{xmin!r} must be an integer multiple of the corresponding chunk size {xcs!r}, but is not."
        assert ymin % ycs == 0, f"{ymin!r} must be an integer multiple of the corresponding chunk size {ycs!r}, but is not."
        assert zmin % zcs == 0, f"{zmin!r} must be an integer multiple of the corresponding chunk size {zcs!r}, but is not."

        grid_coords = [int(xmin/xcs), int(ymin/ycs), int(zmin/zcs)]
        
        return self.compressed_morton_code(grid_coords)

class ShardedIOError(IOError): pass
