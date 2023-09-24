# Copyright (c) 2016, 2017, 2018 Forschungszentrum Juelich GmbH
# Author: Xiao Gui <x.gui@fz-juelich.de>
#
# This software is made available under the MIT licence, see LICENCE.txt.

"""Access to a Neuroglancer sharded pre-computed dataset on the local filesystem.

See the :mod:`~neuroglancer_scripts.accessor` module for a description of the
API.
"""

from collections.abc import Iterator
import neuroglancer_scripts.accessor
import pathlib
import math
import numpy as np
from typing import Dict, Tuple, List, Union, Any
import zlib
import struct
import json
from tempfile import TemporaryDirectory
from uuid import uuid4

_MAX_UINT64 = 0xffffffffffffffff
_VALID_ENCODING = ("gzip", "raw")
_VALID_SHARD_WRITE_STRATEGY = (
    "in memory", # requirement of RAM is higher
    "on disk", # requirement of disk space is higher (both spacce and iinode). Takes longer.
    "append",
)

_ONE = np.uint64(1)

class OnDiskBytesDict(dict):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        self._tmp_dir = pathlib.Path(TemporaryDirectory().name)
        self._tmp_dir.mkdir(parents=True, exist_ok=True)

        self._key_to_filename: Dict[Any, str] = {}
    
    def _delete(self, key):
        filename = self._key_to_filename.pop(key)
        (self._tmp_dir / filename).unlink()
        

    def __setitem__(self, key, value):
        
        assert isinstance(value, bytes), f"Can only set bytes"

        if key in self._key_to_filename:
            self._delete(key)
        
        filename = str(uuid4())
        self._key_to_filename[key] = filename
        with open(self._tmp_dir / filename, "wb") as fp:
            fp.write(value)

    def __getitem__(self, key):
        filename = self._key_to_filename[key]
        with open(self._tmp_dir / filename, "rb") as fp:
            return fp.read()

    def get(self, key):
        return self[key]
        

    def pop(self, key):
        value = self[key]
        self._delete(key)
        return value
        
    def __contains__(self, key):
        return key in self._key_to_filename
    
    def keys(self):
        return self._key_to_filename.keys()
    
    def __len__(self):
        return len(self._key_to_filename)
        

class InMemByteArray(bytearray):
    def __iter__(self) -> Iterator[bytes]:
        yield bytes(self)

class OnDiskByteArray:
    _READ_SIZE = 4096

    def __init__(self) -> None:
        _tmp_dir = pathlib.Path(TemporaryDirectory().name)
        _tmp_dir.mkdir(parents=True, exist_ok=True)
        self._file = _tmp_dir / "sharded_ondisk_bytearray"
        self._len = 0

    def __len__(self):
        return self._len
        
    def __add__(self, o):
        self._len += len(o)
        with open(self._file, "ab") as fp:
            fp.write(o)
        return self

    def __radd__(self, o):
        return self.__add__(o)
    
    def __iter__(self):
        with open(self._file, "rb") as fp:
            while True:
                data = fp.read(self._READ_SIZE)
                if not data:
                    break
                yield data

class MiniShard:
    def __init__(self, combined_key: np.uint64, 
                 minishard_index_encoding,
                 data_encoding,
                 shard_bits,
                 minishard_bits,
                 offset: np.uint64=np.uint64(0),
                 strategy="on disk") -> None:
        assert minishard_index_encoding in _VALID_ENCODING
        assert data_encoding in _VALID_ENCODING
        
        self.shard_bits = shard_bits
        self.minishard_bits = minishard_bits

        self._offset = offset
        self.minishard_index_encoding = minishard_index_encoding
        self.data_encoding = data_encoding

        self.index_encoder = (lambda b: zlib.compress(b)) if minishard_index_encoding == "gzip" else (lambda b: b)
        self.data_encoder = (lambda b: zlib.compress(b)) if data_encoding == "gzip" else (lambda b: b)

        self.combined_key = combined_key

        self._appended = np.uint64(0)
        self._last_chunk_id = np.uint64(0)
        self._chunk_buffer: Dict[np.uint64, bytes] = OnDiskBytesDict() if strategy == "on disk" else dict()

        self.databytearray: Union[OnDiskByteArray, InMemByteArray] = OnDiskByteArray() if strategy == "on disk" else InMemByteArray()
        self.header = np.array([], dtype=np.uint64)
    


    @property
    def offset(self):
        return self._offset
    
    @offset.setter
    def offset(self, val):
        self._offset = np.uint64(val)
        self.header[1] = np.uint64(val)
    
    @property
    def next_cmc(self):
        return np.uint64(self.combined_key + ((_ONE << (self.minishard_bits + self.shard_bits)) * self._appended))

    def store_chunk(self, buf: bytes, cmc: np.uint64):
        chunk_to_store = self.data_encoder(buf)
        if self.can_be_appended(cmc):
            self.append(chunk_to_store, cmc)
            self.flush_buffer()
            return
        self._chunk_buffer[cmc] = chunk_to_store
        

    def append(self, buf: bytes, cmc: np.uint64):
        self.databytearray += buf
        
        new_chunk_id = cmc - self._last_chunk_id
        self._last_chunk_id = cmc
        self.header = np.append(self.header, np.uint64(new_chunk_id))
        offset = self.offset if self._appended == np.uint64(0) else np.uint64(0) # chunks are contiguous, so offset is always 0 (except for first append)
        self.header = np.append(self.header, offset)
        self.header = np.append(self.header, np.uint64(len(buf)))

        self._appended += np.uint64(1)

    
    def can_be_appended(self, cmc: np.uint64):
        """Check if the compressed-morton-code is ready to be appended"""
        
        if self.next_cmc > cmc:
            import pdb
            pdb.set_trace()
            raise RuntimeError(f"cmc {cmc} < next_cmc {self.next_cmc}")
        return self.next_cmc == cmc
    
    def flush_buffer(self):
        """In the event that """
        while self.next_cmc in self._chunk_buffer:
            buffer = self._chunk_buffer.pop(self.next_cmc)
            self.append(buffer, self.next_cmc)
        
        if any(key < self.next_cmc for key in self._chunk_buffer.keys()):
            raise RuntimeError(f"Key exist that is less than id to check")

    def close(self):
        while len(self._chunk_buffer) > 0:
            self.append(b'', self.next_cmc)
            self.flush_buffer()


class Shard:
    def __init__(self, base_dir, shard_bits: int, shard_key: np.uint64, minishard_bits:np.uint64,
                 minishard_index_encoding,
                 data_encoding,
                 preshift_bits=None,):
        self.preshift_bits = preshift_bits
        shard_key_str = hex(shard_key)[2:].rjust(math.ceil(shard_bits / 4), "0")
        
        self.file_path = pathlib.Path(base_dir) / f"{shard_key_str}.shard"
        self.shard_key = shard_key
        self._minishard_bits = minishard_bits
        self.shard_bits = shard_bits
        self.minishard_mask = ~np.uint64(_MAX_UINT64 >> self.minishard_bits << self.minishard_bits)
        self.minishard_dict: Dict[np.uint64, MiniShard] = {}

        self.minishard_index_encoding = minishard_index_encoding
        self.data_encoding = data_encoding
    
    @property
    def minishard_bits(self):
        """Readonly once set"""
        return self._minishard_bits
    
    
    def store_chunk(self, buf: bytes, cmc: np.uint64):
        minishard_key = self.get_minishard_key(cmc)
        if minishard_key not in self.minishard_dict:
            self.minishard_dict[minishard_key] = MiniShard((self.shard_key << self.minishard_bits) + minishard_key,
                                                           minishard_bits=self.minishard_bits,
                                                           shard_bits=self.shard_bits,
                                                           minishard_index_encoding=self.minishard_index_encoding,
                                                           data_encoding=self.data_encoding)
        self.minishard_dict[minishard_key].store_chunk(buf, cmc)

    def get_minishard_key(self, cmc: np.uint64) -> np.uint64:
        return (self.minishard_mask & cmc)
    
    def close(self):
        self.file_path.parent.mkdir(exist_ok=True, parents=True)
        with open(self.file_path, "wb") as fp:
            fp.write(b"\0"*int((2**self.minishard_bits) * 16))
            shard_index_ba = bytearray()

            data_size = 0
            for minishard in self.minishard_dict.values():
                minishard.close()

                for b in minishard.databytearray:
                    fp.write(b)

                minishard.offset = data_size
                data_size += len(minishard.databytearray)
                del minishard.databytearray

            shard_size_tally = 0
            for minishard in self.minishard_dict.values():
                # turning [0, 1, 2, 3, 4, 5] into [0, 3, 1, 4, 2, 5]
                header_bytes = np.reshape(minishard.header, (3, int(len(minishard.header) / 3)), order="F").tobytes(order="C")
                
                header_bytes = minishard.index_encoder(header_bytes)
                fp.write(header_bytes)

                shard_index_ba += struct.pack("<Q", data_size + shard_size_tally)
                
                shard_size_tally += len(header_bytes)
                shard_index_ba += struct.pack("<Q", data_size + shard_size_tally)
                
            
            if len(shard_index_ba) != (2 ** self.minishard_bits) * 16:
                print(f"Writing shard index error! Expected {(2 ** self.minishard_bits) * 16} bytes, got {len(shard_index_ba)}")
                assert len(shard_index_ba) < (2 ** self.minishard_bits) * 16, f"len(shard_index_ba) is larger? this shouldn't happend!"
                while len(shard_index_ba) < (2 ** self.minishard_bits) * 16:
                    shard_index_ba += struct.pack("<Q", data_size + shard_size_tally)
                    shard_index_ba += struct.pack("<Q", data_size + shard_size_tally)
                    
            fp.seek(0)
            fp.write(bytes(shard_index_ba))

class ShardedScale:
    def __init__(self, base_dir, key, chunk_sizes,
                 preshift_bits=0,
                 hash="identity",
                 minishard_bits=2,
                 shard_bits=2,
                 minishard_index_encoding="gzip",
                 data_encoding="gzip",
                 mip_sizes=None):
        
        self.chunk_sizes = chunk_sizes
        self.mip_sizes = mip_sizes
        self.grid_sizes = [math.ceil(mipsize / chunksize) for mipsize, chunksize in zip(self.mip_sizes, self.chunk_sizes)]

        assert self.chunk_sizes and len(self.chunk_sizes) == 3 and all(isinstance(v, int) for v in chunk_sizes), f"chunk_sizes must be defined, len 3 and all int. It represents x, y, z dimensions."
        assert self.grid_sizes and len(self.grid_sizes) == 3 and all(isinstance(v, int) for v in self.grid_sizes), f"grid_sizes must be defined, and must be len 3 all int (should be math.ceil(sizes / chunk_sizes))"

        self.base_dir = pathlib.Path(base_dir) / key
        self.preshift_bits = preshift_bits

        self.num_bits = [math.ceil(math.log2(grid_size)) for grid_size in self.grid_sizes]
        assert sum(self.num_bits) <= 64, f"Cannot use sharded file accessor for self.grid_sizes {self.grid_sizes}. It requires {self.num_bits}, larger than the max possible, 64."

        if self.preshift_bits is None:
            self.preshift_bits = np.uint64(64 - np.sum(self.num_bits))

        self.hash = hash
        assert self.hash == "identity"

        assert minishard_bits >= 0
        self._minishard_bits = np.uint64(minishard_bits)

        assert shard_bits >= 0
        self._shard_bits = np.uint64(shard_bits)
        self.minishard_index_encoding = minishard_index_encoding
        self.data_encoding = data_encoding

        self.minishard_mask = ~(np.uint64(_MAX_UINT64) >> self.minishard_bits << self.minishard_bits)
        
        movement = self.minishard_bits + self.shard_bits
        self.shard_mask = ~(np.uint64(_MAX_UINT64) >> movement << movement)
        self.shard_mask = self.shard_mask & (~self.minishard_mask)
        self.shard_dict: Dict[np.uint64, Shard] = {}
    
    @property
    def shard_bits(self):
        """Readonly after setting on init"""
        return self._shard_bits
    
    @property
    def minishard_bits(self):
        return self._minishard_bits

    def store_chunk(self, buf, chunk_coords):
        
        cmc = self.compressed_morton_code(chunk_coords)
        shard_key = self.get_shard_key(cmc)
        if shard_key not in self.shard_dict:
            self.shard_dict[shard_key] = Shard(self.base_dir, self.shard_bits, shard_key, self.minishard_bits,
                                               minishard_index_encoding=self.minishard_index_encoding,
                                               data_encoding=self.data_encoding)
        self.shard_dict[shard_key].store_chunk(buf, cmc)

    def compressed_morton_code(self, chunk_coords):
        xmin, xmax, ymin, ymax, zmin, zmax = chunk_coords
        xcs, ycs, zcs = self.chunk_sizes
        assert xmin % xcs == 0, f"{xmin!r} must be an integer multiple of the corresponding chunk size {xcs!r}, but is not."
        assert ymin % ycs == 0, f"{ymin!r} must be an integer multiple of the corresponding chunk size {ycs!r}, but is not."
        assert zmin % zcs == 0, f"{zmin!r} must be an integer multiple of the corresponding chunk size {zcs!r}, but is not."

        grid_coords = [int(xmin/xcs), int(ymin/ycs), int(zmin/zcs)]
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
    
    def get_shard_key(self, cmc: np.uint64) -> np.uint64:
        return (self.shard_mask & cmc) >> self.minishard_bits
    
    def to_json(self):
        return {
            "@type": "neuroglancer_uint64_sharded_v1",
            "preshift_bits": int(self.preshift_bits),
            "hash": self.hash,
            "minishard_bits": int(self.minishard_bits),
            "shard_bits": int(self.shard_bits),
            "minishard_index_encoding": self.minishard_index_encoding,
            "data_encoding": self.data_encoding
        }
    
    def close(self):
        for shard in self.shard_dict.values():
            shard.close()



class ShardedFileAccessor(neuroglancer_scripts.accessor.Accessor):
    """Access Neuroglancer sharded precomputed pyramid on the local file system.
    
    :param str base_dir: path to the directory containing the pyramid
    :param int preshift_bits: 
    :param str hash: 
    :param str minishard_bits: 
    :param str shard_bits: 
    :param str minishard_index_encoding: 
    :param str data_encoding: 
    :param List[int] sizes:
    """
    can_read = False
    can_write = True

    def __init__(self, base_dir, key_to_mip_sizes: Dict[str, Union[List[int], Tuple[int]]]) -> None:
        self.base_dir = pathlib.Path(base_dir)
        self.shard_dict: Dict[ str, ShardedScale ] = {}
        assert (
            key_to_mip_sizes
            and isinstance(key_to_mip_sizes, dict)
            and all(
                (    
                    len(value) == 3
                    and all(isinstance(v, int) for v in value)
                )
                for value in key_to_mip_sizes.values()
            )
        )
        self.key_to_mip_sizes = key_to_mip_sizes
        "1048576"

    def file_exists(self, relative_path: str):
        return (self.base_dir / relative_path).exists()
    
    def fetch_file(self, relative_path):
        with open(self.base_dir / relative_path, "rb") as fp:
            return fp.read()
    
    def store_file(self, relative_path, buf, overwrite=False, **kwargs):
        if not overwrite and self.file_exists(relative_path):
            raise IOError(f"file at {relative_path} already exists")
        with open(self.base_dir / relative_path, "wb") as fp:
            fp.write(buf)
    
    def fetch_chunk(self, key, chunk_coords):
        raise NotImplementedError
    
    def store_chunk(self, buf, key, chunk_coords, **kwargs):
        assert key in self.key_to_mip_sizes, f"Expecting key {key} in key_to_mip_sizes, but were not: {list(self.key_to_mip_sizes.keys())}"
        if key not in self.shard_dict:
            self.shard_dict[key] = ShardedScale(base_dir=self.base_dir, key=key, mip_sizes=self.key_to_mip_sizes[key], chunk_sizes=(64, 64, 64))
        self.shard_dict[key].store_chunk(buf, chunk_coords)

    def close(self):
        scale_arr = []
        for scale in self.shard_dict.values():
            scale.close()
            scale_arr.append(scale.to_json())
        
        scale_str = json.dumps(scale_arr)
        self.store_file("segments.json", scale_str.encode("utf-8"), overwrite=True)