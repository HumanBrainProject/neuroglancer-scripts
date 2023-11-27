from typing import List, Dict, Any
from abc import ABC, abstractmethod
import math
import numpy as np
import zlib

_MAX_UINT64 = 0xffffffffffffffff

# spec from https://github.com/google/neuroglancer/blob/master/src/neuroglancer/datasource/precomputed/sharded.md # noqa

_VALID_ENCODING = ("gzip", "raw")

CHUNK_COORD_ERROR_MSG = (
    "{value} must be an integer multiple of the corresponding chunk size"
    " {chunk_size}, but is not."
)


class ShardSpec:
    def __init__(self, minishard_bits: int, shard_bits: int,
                 hash: str = "identity",
                 minishard_index_encoding: str = "raw",
                 data_encoding: str = "raw",
                 preshift_bits: int = 0):

        try:
            assert minishard_bits >= 0, "minishard_bits must be >= 0"
            assert shard_bits >= 0, "shard_bits must be >= 0"
            assert hash == "identity", (
                "Only identity hash is supported at the moment"
            )
            assert preshift_bits >= 0, "preshift_bits needs to be >= 0"
            assert data_encoding in _VALID_ENCODING
            assert minishard_index_encoding in _VALID_ENCODING
        except AssertionError as e:
            raise ShardedIOError from e

        self.minishard_bits = np.uint64(minishard_bits)
        self.shard_bits = np.uint64(shard_bits)
        self.hash = hash
        self.minishard_index_encoding = minishard_index_encoding
        self.data_encoding = data_encoding
        self.preshift_bits = np.uint64(preshift_bits)

        self._shard_mask = None
        self._minishard_mask = None
        self._preshift_mask = None

    def data_encoder(self, b: bytes) -> bytes:
        return zlib.compress(b) if self.data_encoding == "gzip" else b

    def data_decoder(self, b: bytes) -> bytes:
        return zlib.decompress(b) if self.data_encoding == "gzip" else b

    def index_encoder(self, b: bytes) -> bytes:
        return (
            zlib.compress(b)
            if self.minishard_index_encoding == "gzip"
            else b
        )

    def index_decoder(self, b: bytes) -> bytes:
        return (
            zlib.decompress(b)
            if self.minishard_index_encoding == "gzip"
            else b
        )

    def id_hash(self, cmc: np.uint64) -> np.uint64:
        # murmurhash3_x86_128 not yet supported
        return cmc

    def to_dict(self):
        return {
            "@type": "neuroglancer_uint64_sharded_v1",
            "minishard_bits": int(self.minishard_bits),
            "shard_bits": int(self.shard_bits),
            "hash": self.hash,
            "minishard_index_encoding": self.minishard_index_encoding,
            "data_encoding": self.data_encoding,
            "preshift_bits": int(self.preshift_bits),
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
            self._minishard_mask = ~(
                np.uint64(_MAX_UINT64) >> movement << movement
            )
        return self._minishard_mask

    @property
    def preshift_mask(self) -> np.uint64:
        if not self._preshift_mask:
            movement = np.uint64(self.preshift_bits)
            self._preshift_mask = ~(
                np.uint64(_MAX_UINT64) >> movement << movement
            )
        return self._preshift_mask


class ShardVolumeSpec:
    def __init__(self, chunk_sizes: List[int], sizes: List[int]):
        if not (
            sizes
            and len(sizes) == 3
            and all((
                isinstance(v, int) and v > 0
            ) for v in sizes)
        ):
            raise ShardedIOError("sizes must be defined, and must be len 3 all"
                                 " must int and > 0")
        self.sizes = sizes

        if not (
            chunk_sizes
            and len(chunk_sizes) == 3
            and all((
                isinstance(v, int)
                and v > 0
            ) for v in chunk_sizes)
            and len({cz for cz in chunk_sizes}) == 1
        ):
            raise ShardedIOError("chunk_sizes must be defined, len 3 and all "
                                 "int. chunk_sizes must be same in all "
                                 f"dimensions.: {chunk_sizes}")

        self.chunk_sizes = chunk_sizes
        self.grid_sizes = [math.ceil(size/chunk_size)
                           for chunk_size, size
                           in zip(self.chunk_sizes, self.sizes)]
        self.num_bits = [math.ceil(math.log2(grid_size))
                         for grid_size in self.grid_sizes]

        if sum(self.num_bits) > 64:
            raise ShardedIOError(f"Cannot use sharded file accessor for "
                                 f"self.grid_sizes {self.grid_sizes}. It "
                                 f"requires {self.num_bits}, larger than the "
                                 "max possible, 64.")

    def compressed_morton_code(self, grid_coords: List[int]):
        if any(
            not isinstance(grid_coord, int) or grid_coord < 0
            for grid_coord in grid_coords
        ):
            raise ShardedIOError(f"{grid_coords!r} needs to be int, and >= 0")
        if not all(
            grid_coord <= grid_size
            for grid_coord, grid_size in zip(grid_coords, self.grid_sizes)
        ):
            raise ShardedIOError(f"{grid_coords!r} must be element-wise less "
                                 "or eq to {self.grid_sizes!r}, but is not")

        j = np.uint64(0)
        one = np.uint64(1)
        code = np.uint64(0)

        for i in range(max(self.num_bits)):
            for dim in range(3):
                if 2 ** i < self.grid_sizes[dim]:
                    bit = (
                        ((np.uint64(grid_coords[dim]) >> np.uint64(i)) & one)
                        << j)
                    code |= bit
                    j += one
        return code

    def get_cmc(self, chunk_coords: List[int]) -> np.uint64:
        xmin, xmax, ymin, ymax, zmin, zmax = chunk_coords
        xcs, ycs, zcs = self.chunk_sizes

        if xmin % xcs != 0:
            raise ShardedIOError(
                CHUNK_COORD_ERROR_MSG.format(value=xmin, chunk_size=xcs)
            )
        if ymin % ycs != 0:
            raise ShardedIOError(
                CHUNK_COORD_ERROR_MSG.format(value=ymin, chunk_size=ycs)
            )
        if zmin % zcs != 0:
            raise ShardedIOError(
                CHUNK_COORD_ERROR_MSG.format(value=zmin, chunk_size=zcs)
            )

        grid_coords = [int(xmin/xcs), int(ymin/ycs), int(zmin/zcs)]

        return self.compressed_morton_code(grid_coords)


class CMCReadWrite(ABC):

    can_read_cmc = False
    can_write_cmc = False

    def __init__(self, shard_spec: ShardSpec) -> None:
        self.shard_spec = shard_spec
        self.header_byte_length = int(2 ** self.shard_spec.minishard_bits * 16)

    def fetch_cmc_chunk(self, cmc: np.uint64):
        raise NotImplementedError

    def store_cmc_chunk(self, buf: bytes, cmc: np.uint64):
        raise NotImplementedError

    def close(self):
        pass

    def _hash(self, cmc: np.uint64):
        return self.shard_spec.id_hash(cmc >> self.shard_spec.preshift_bits)

    def get_shard_key(self, cmc: np.uint64) -> np.uint64:
        return (
            (self.shard_spec.shard_mask & self._hash(cmc))
            >> self.shard_spec.minishard_bits
        )

    def get_minishard_key(self, cmc: np.uint64) -> np.uint64:
        return (self.shard_spec.minishard_mask & self._hash(cmc))

    def read_bytes(self, offset: int, length: int) -> bytes:
        raise NotImplementedError


class ShardCMC(CMCReadWrite, ABC):

    def __init__(self, shard_key: np.uint64, shard_spec: ShardSpec) -> None:
        super().__init__(shard_spec)

        self.shard_key_str = hex(shard_key)[2:].rjust(
            math.ceil(self.shard_spec.shard_bits / 4), "0"
        )
        self.shard_key = shard_key
        self.minishard_dict: Dict[np.uint64, CMCReadWrite] = {}
        self.ro_minishard_dict: Dict[np.uint64, CMCReadWrite] = {}

        self.can_read_cmc = False

        if self.file_exists(f"{self.shard_key_str}.shard"):
            self.is_legacy = False
            self.can_read_cmc = True
        elif (
            self.file_exists(f"{self.shard_key_str}.index")
            and self.file_exists(f"{self.shard_key_str}.data")
        ):
            self.is_legacy = True
            self.can_read_cmc = True

    def populate_minishard_dict(self):

        if self.can_read_cmc:
            offsets = self.get_minishards_offsets()
            for offset, end in zip(offsets[::2], offsets[1::2]):
                start = int(offset + self.header_byte_length)
                length = int(end - offset)
                minishard_raw_buffer = self.read_bytes(start, length)
                minishard_decoded_buffer = self.shard_spec.index_decoder(
                    minishard_raw_buffer)

                minishard = ReadableMiniShardCMC(self,
                                                 minishard_decoded_buffer)
                first_cmc = minishard.minishard_index[0]
                minishard_key = self.get_minishard_key(first_cmc)
                self.ro_minishard_dict[minishard_key] = minishard

    def get_minishards_offsets(self):
        assert self.can_read_cmc
        header_content = self.read_bytes(0, self.header_byte_length)
        return np.frombuffer(header_content, dtype=np.uint64)

    @abstractmethod
    def file_exists(self, filepath) -> bool:
        raise NotImplementedError


class ReadableMiniShardCMC(CMCReadWrite):

    can_read_cmc = True
    can_write_cmc = False

    def __init__(self, parent_shard: "CMCReadWrite", header_buffer: bytes):
        assert parent_shard.can_read_cmc
        super().__init__(parent_shard.shard_spec)

        self.parent_shard = parent_shard
        self.minishard_index = np.frombuffer(header_buffer, dtype=np.uint64)
        if len(self.minishard_index) % 3 != 0:
            raise ShardedIOError("self.minishard_index should be divisible by"
                                 f" 3, but {len(self.minishard_index)} is not")

        self.num_chunks = int(len(self.minishard_index) / 3)

    def fetch_cmc_chunk(self, cmc: np.uint64):
        idx_tally = self.minishard_index[0]
        chunk_idx = 0

        while idx_tally < cmc:
            chunk_idx += 1
            idx_tally += self.minishard_index[chunk_idx]
        if idx_tally != cmc:
            raise ShardedIOError(f"Expecting sum of first {chunk_idx} to equal"
                                 f" {cmc}, but did not: {idx_tally}")

        byte_length = self.minishard_index[2 * self.num_chunks + chunk_idx]

        # start at end of header
        byte_offset = self.parent_shard.header_byte_length
        # sum of all delta offsets (self inclusive)
        start, end = self.num_chunks, self.num_chunks + chunk_idx + 1
        byte_offset += np.sum(
            self.minishard_index[start:end]
        )
        # sum of all PREVIOUS bytelength (self non-inclusive)
        start, end = 2 * self.num_chunks, 2 * self.num_chunks + chunk_idx
        byte_offset += np.sum(
            self.minishard_index[start:end]
        )
        buf = self.parent_shard.read_bytes(int(byte_offset), int(byte_length))
        return self.shard_spec.data_decoder(buf)


class ShardedScaleBase(CMCReadWrite, ABC):

    def __init__(self, key, shard_spec: ShardSpec,
                 shard_volume_spec: ShardVolumeSpec):
        CMCReadWrite.__init__(self, shard_spec)

        self.key = key
        self.shard_volume_spec = shard_volume_spec

    @abstractmethod
    def get_shard(self, shard_key: np.uint64) -> CMCReadWrite:
        raise NotImplementedError

    def store_cmc_chunk(self, buf: bytes, cmc: np.uint64):
        shard_key = self.get_shard_key(cmc)
        shard = self.get_shard(shard_key)
        assert shard.can_write_cmc
        return shard.store_cmc_chunk(buf, cmc)

    def store_chunk(self, buf, chunk_coords, **kwargs):
        cmc = self.shard_volume_spec.get_cmc(chunk_coords)
        return self.store_cmc_chunk(buf, cmc)

    def fetch_cmc_chunk(self, cmc: np.uint64):
        shard_key = self.get_shard_key(cmc)
        shard = self.get_shard(shard_key)
        assert shard.can_read_cmc
        return shard.fetch_cmc_chunk(cmc)

    def fetch_chunk(self, chunk_coords):
        cmc = self.shard_volume_spec.get_cmc(chunk_coords)
        return self.fetch_cmc_chunk(cmc)

    def to_json(self):
        return {
            "@type": "neuroglancer_uint64_sharded_v1",
            **self.shard_spec.to_dict()
        }

    @staticmethod
    def is_sharded(scale: Dict[str, Any]) -> bool:
        try:
            return (scale["sharding"]["@type"]
                    == "neuroglancer_uint64_sharded_v1")
        except Exception:
            return False


class ShardedAccessorBase(ABC):

    @property
    def info(self):
        if hasattr(self, "_info"):
            return self._info
        raise AttributeError("Info not yet defined")

    @info.setter
    def info(self, val):
        assert isinstance(val, dict), ".info must be a dictionary"
        assert val.get("scales"), ".info must have scales property"
        for scale in val.get("scales"):
            key = scale.get("key")
            if not scale.get("sharding"):
                raise ShardedIOError(f"Scale with key {key} is not a"
                                     " sharded source")
            sharding = scale.get("sharding")
            shard_type = sharding.pop("@type", None)
            if shard_type != "neuroglancer_uint64_sharded_v1":
                raise ShardedIOError("Shard spec does not have key "
                                     "'neuroglancer_uint64_sharded_v1'."
                                     f"It is instead {shard_type!r}")
        self._info = val

    def get_scale(self, key) -> Dict[str, Any]:
        scales = self.info.get("scales")
        try:
            scale, = [scale for scale in scales if scale.get("key") == key]
            return scale
        except ValueError as e:
            raise ValueError(f"key {key!r} not found in scales. Possible "
                             "values are"
                             ', '.join([scale.get('key') for scale in scales])
                             ) from e

    def get_sharding_spec(self, key):
        scale = self.get_scale(key)
        sharding = scale.get("sharding")
        sharding.pop("@type", None)
        return sharding

    @staticmethod
    def info_is_sharded(info_json):
        scales = info_json.get("scales", [])
        return (
            len(scales) > 0
            and all(ShardedScaleBase.is_sharded(s) for s in scales)
        )


class ShardedIOError(IOError):
    ...
