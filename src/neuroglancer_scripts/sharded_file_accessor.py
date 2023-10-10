"""Access to a Neuroglancer sharded pre-computed dataset on the local filesystem.

See the :mod:`~neuroglancer_scripts.accessor` module for a description of the
API.
"""

from collections.abc import Iterator
import neuroglancer_scripts.accessor
from neuroglancer_scripts.sharded_base import (
    ShardSpec, CMCReadWrite, ShardedScaleBase, ReadableMiniShardCMC, ShardedAccessorBase, ShardVolumeSpec, ShardedIOError
)
import pathlib
import math
import numpy as np
from typing import Dict, List, Union, Any
import struct
import json
from tempfile import TemporaryDirectory
from uuid import uuid4


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


class MiniShard(CMCReadWrite):
    
    can_read_cmc = False
    can_write_cmc = True

    def __init__(self, shard_spec: ShardSpec,
                 offset: np.uint64=np.uint64(0),
                 strategy="on disk") -> None:
        super().__init__(shard_spec)
        self._offset = offset

        self._appended = np.uint64(0)

        self._last_chunk_id = np.uint64(0)
        self._chunk_buffer: Dict[np.uint64, bytes] = OnDiskBytesDict() if strategy == "on disk" else dict()

        self.databytearray: Union[OnDiskByteArray, InMemByteArray] = OnDiskByteArray() if strategy == "on disk" else InMemByteArray()
        self.header = np.array([], dtype=np.uint64)

        self.masked_bits = None

    @property
    def next_cmc(self) -> np.uint64:
        next_val = self._appended
        return (
            (
                next_val >> self.shard_spec.preshift_bits
                << (self.shard_spec.preshift_bits + self.shard_spec.shard_bits + self.shard_spec.minishard_bits)
            )
            + self.masked_bits
            + (next_val & self.shard_spec.preshift_mask)
        )

    @property
    def offset(self):
        return self._offset
    
    @offset.setter
    def offset(self, val):
        self._offset = np.uint64(val)
        self.header[1] = np.uint64(val)
    
    def store_cmc_chunk(self, buf: bytes, cmc: np.uint64):
        if not self.masked_bits:
            self.masked_bits = (
                (self.shard_spec.minishard_mask + self.shard_spec.shard_mask)
                << np.uint64(self.shard_spec.preshift_bits)
            ) & cmc
            
        chunk_to_store = self.shard_spec.data_encoder(buf)
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
            raise RuntimeError(f"cmc {cmc} < next_cmc {self.next_cmc}")
        return self.next_cmc == cmc
    
    def flush_buffer(self):
        """In the event that """
        while self.next_cmc in self._chunk_buffer:
            buffer = self._chunk_buffer.pop(self.next_cmc)
            self.append(buffer, self.next_cmc)
        
        if any(key < self.next_cmc for key in self._chunk_buffer.keys()):
            raise RuntimeError(f"Key exist that is less than id to check {self._chunk_buffer.keys()}, {self.next_cmc}")

    def close(self):
        while len(self._chunk_buffer) > 0:
            self.append(b'', self.next_cmc)
            self.flush_buffer()


class Shard(CMCReadWrite):

    # legacy shard has two files, .index and .data
    # latest implementation is the concatentation of [.index, .data] into .shard
    is_legacy = False

    can_read_cmc = False
    can_write_cmc = True

    def __init__(self, base_dir, shard_key: np.uint64, shard_spec:ShardSpec):
        super().__init__(shard_spec)
        
        shard_key_str = hex(shard_key)[2:].rjust(math.ceil(self.shard_spec.shard_bits / 4), "0")
        
        self.file_path = pathlib.Path(base_dir) / f"{shard_key_str}.shard"
        self.shard_key = shard_key
        self.minishard_dict: Dict[np.uint64, CMCReadWrite] = {}

        if self.file_path.is_file():
            self.can_write_cmc = False
            self.can_read_cmc = True

        if self.file_path.with_suffix(".index").is_file():
            self.can_write_cmc = False
            self.can_read_cmc = True
            self.is_legacy = True

        if self.can_read_cmc:
            offsets = self.get_minishards_offsets()
            for offset, end in zip(offsets[::2], offsets[1::2]):
                minishard_raw_buffer = self.read_bytes(int(offset + self.header_byte_length), int(end - offset))
                minishard_decoded_buffer = self.shard_spec.index_decoder(minishard_raw_buffer)

                minishard = ReadableMiniShardCMC(self, minishard_decoded_buffer, self.shard_spec)
                first_cmc = minishard.minishard_index[0]
                minishard_key = self.get_minishard_key(first_cmc)
                self.minishard_dict[minishard_key] = minishard

    def read_bytes(self, offset: int, length: int) -> bytes:
        if not self.can_read_cmc:
            raise ShardedIOError("Writeonly shard")

        file_path = self.file_path
        if self.is_legacy:
            if offset < self.header_byte_length:
                file_path = file_path.with_suffix(".index")
            else:
                file_path = file_path.with_suffix(".data")
                offset = offset - self.header_byte_length
            
        with open(file_path, "rb") as fp:
            fp.seek(offset)
            return fp.read(length)
    
    def store_cmc_chunk(self, buf: bytes, cmc: np.uint64):
        if not self.can_write_cmc:
            raise ShardedIOError(f"Readonly shard")
        
        minishard_key = self.get_minishard_key(cmc)
        if minishard_key not in self.minishard_dict:
            self.minishard_dict[minishard_key] = MiniShard(self.shard_spec)
        self.minishard_dict[minishard_key].store_cmc_chunk(buf, cmc)

    def fetch_cmc_chunk(self, cmc: np.uint64):
        minishard_key = self.get_minishard_key(cmc)
        assert minishard_key in self.minishard_dict
        return self.minishard_dict[minishard_key].fetch_cmc_chunk(cmc)
    
    def close(self):
        self.file_path.parent.mkdir(exist_ok=True, parents=True)
        with open(self.file_path, "wb") as fp:
            fp.write(b"\0"*int((2**self.shard_spec.minishard_bits) * 16))
            shard_index_ba = bytearray()

            # minishard order must always monotoneously increasing
            # by default, python dict iter respects order of insertion
            sorted_mini_dict: List[MiniShard] = [self.minishard_dict[key] for key in sorted(self.minishard_dict.keys())]
            assert all(isinstance(minishard, MiniShard) for minishard in sorted_mini_dict)
            data_size = 0
            for minishard in sorted_mini_dict:
                minishard.close()

                for b in minishard.databytearray:
                    fp.write(b)

                minishard.offset = data_size
                data_size += len(minishard.databytearray)
                del minishard.databytearray

            shard_size_tally = 0
            for minishard in sorted_mini_dict:
                # turning [0, 1, 2, 3, 4, 5] into [0, 3, 1, 4, 2, 5]
                header_byte_buffer = np.reshape(minishard.header, (3, int(len(minishard.header) / 3)), order="F").tobytes(order="C")
                
                header_byte_buffer = self.shard_spec.index_encoder(header_byte_buffer)
                fp.write(header_byte_buffer)

                shard_index_ba += struct.pack("<Q", data_size + shard_size_tally)
                
                shard_size_tally += len(header_byte_buffer)
                shard_index_ba += struct.pack("<Q", data_size + shard_size_tally)
            
            if len(shard_index_ba) != (2 ** self.shard_spec.minishard_bits) * 16:
                print(f"Writing shard index error! Expected {(2 ** self.shard_spec.minishard_bits) * 16} bytes, got {len(shard_index_ba)}")
                assert len(shard_index_ba) < (2 ** self.shard_spec.minishard_bits) * 16, f"len(shard_index_ba) is larger? this shouldn't happend!"
                while len(shard_index_ba) < (2 ** self.shard_spec.minishard_bits) * 16:
                    shard_index_ba += struct.pack("<Q", data_size + shard_size_tally)
                    shard_index_ba += struct.pack("<Q", data_size + shard_size_tally)
                    
            fp.seek(0)
            fp.write(bytes(shard_index_ba))


class ShardedScale(ShardedScaleBase):
    
    can_read_cmc = True
    can_write_cmc = True

    def __init__(self, base_dir, key: str, 
                 shard_spec: ShardSpec,
                 shard_volume_spec: ShardVolumeSpec):
        super().__init__(key, shard_spec, shard_volume_spec)
        self.base_dir = pathlib.Path(base_dir) / key
        self.shard_dict: Dict[np.uint64, Shard] = {}
    
    def get_shard(self, shard_key: np.uint64):
        if shard_key not in self.shard_dict:
            self.shard_dict[shard_key] = Shard(self.base_dir, shard_key, self.shard_spec)
        return self.shard_dict[shard_key]

    def close(self):
        for shard in self.shard_dict.values():
            shard.close()

class ShardedFileAccessor(neuroglancer_scripts.accessor.Accessor, ShardedAccessorBase):
    """Access Neuroglancer sharded precomputed pyramid on the local file system.
    
    :param str base_dir: path to the directory containing the pyramid
    :param dict key_to_mip_sizes: 
    """
    can_read = False
    can_write = False

    def __init__(self, base_dir, shard_volume_spec_dict: Dict[str, ShardVolumeSpec]=None) -> None:
        ShardedAccessorBase.__init__(self)
        self.base_dir = pathlib.Path(base_dir)
        self.shard_dict: Dict[ str, ShardedScale ] = {}

        if shard_volume_spec_dict:
            self.shard_volume_spec_dict = shard_volume_spec_dict
            self.can_write = True
        
        try:
            self.info = json.loads(self.fetch_file("info"))
            self.can_read = True
        except IOError:
            ...
    
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
        if key not in self.shard_dict:
            sharding = self.get_sharding_spec(key)    
            chunk_sizes, = self.get_scale(key).get("chunk_sizes", [[]])
            size = self.get_scale(key).get("size", [])
            shard_spec = ShardSpec(**sharding)
            shard_volume_spec = ShardVolumeSpec(chunk_sizes, size)
            self.shard_dict[key] = ShardedScale(base_dir=self.base_dir, key=key, shard_spec=shard_spec, shard_volume_spec=shard_volume_spec)
        return self.shard_dict[key].fetch_chunk(chunk_coords)
    
    def store_chunk(self, buf, key, chunk_coords, **kwargs):
        assert key in self.shard_volume_spec_dict, f"Expecting key {key} in key_to_mip_sizes, but were not: {list(self.shard_volume_spec_dict.keys())}"
        if key not in self.shard_dict:
            shard_volume_spec = self.shard_volume_spec_dict[key]
            shard_spec = shard_volume_spec.generate_shard_spec()
            self.shard_dict[key] = ShardedScale(base_dir=self.base_dir, key=key, shard_spec=shard_spec, shard_volume_spec=self.shard_volume_spec_dict[key])
        self.shard_dict[key].store_chunk(buf, chunk_coords)

    def close(self):
        scale_arr = []
        for scale in self.shard_dict.values():
            scale.close()
            scale_arr.append(scale.to_json())
        
        scale_str = json.dumps(scale_arr)
        self.store_file("segments.json", scale_str.encode("utf-8"), overwrite=True)
