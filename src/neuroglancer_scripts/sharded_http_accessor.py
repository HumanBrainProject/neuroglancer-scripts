from typing import Dict
import numpy as np
import math
import json

from neuroglancer_scripts.sharded_base import (
    ShardSpec,
    CMCReadWrite,
    ShardedScaleBase,
    ReadableMiniShardCMC,
    ShardedAccessorBase,
    ShardVolumeSpec,
    ShardedIOError,
)
import neuroglancer_scripts.http_accessor
import requests


class HttpShard(CMCReadWrite):

    # legacy shard has two files, .index and .data
    # latest implementation is the concatentation of [.index,.data] into .shard
    is_legacy = False

    can_read_cmc = True
    can_write_cmc = False

    def __init__(self, base_url, session: requests.Session,
                 shard_key: np.uint64,
                 shard_spec: ShardSpec):
        super().__init__(shard_spec)

        self.shard_key_str = hex(shard_key)[2:].rjust(
            math.ceil(self.shard_spec.shard_bits / 4), "0"
        )

        self.base_url = base_url
        self._session = session
        self.shard_key = shard_key
        self.minishard_dict: Dict[np.uint64, CMCReadWrite] = {}

        if (
            self.file_exists(f"{self.shard_key_str}.index")
            and self.file_exists(f"{self.shard_key_str}.data")
        ):
            self.is_legacy = True
        else:
            assert self.file_exists(f"{self.shard_key_str}.shard")

        offsets = self.get_minishards_offsets()
        for offset, end in zip(offsets[::2], offsets[1::2]):
            start = int(offset + self.header_byte_length)
            length = int(end - offset)
            minishard_raw_buffer = self.read_bytes(start, length)
            minishard_decoded_buffer = self.shard_spec.index_decoder(
                minishard_raw_buffer)

            minishard = ReadableMiniShardCMC(self, minishard_decoded_buffer,
                                             self.shard_spec)
            first_cmc = minishard.minishard_index[0]
            minishard_key = self.get_minishard_key(first_cmc)
            self.minishard_dict[minishard_key] = minishard

    def file_exists(self, filepath):
        resp = self._session.get(f"{self.base_url}{filepath}")
        if resp.status_code in (200, 404):
            return resp.status_code == 200
        resp.raise_for_status()
        return False

    def read_bytes(self, offset: int, length: int) -> bytes:
        if not self.can_read_cmc:
            raise ShardedIOError("Shard cannot read")
        self._session.get(f"{self.base_url}")
        file_url = f"{self.base_url}{self.shard_key_str}"
        if self.is_legacy:
            if offset < self.header_byte_length:
                file_url += ".index"
            else:
                file_url += ".data"
                offset = offset - self.header_byte_length
        else:
            file_url += ".shard"

        range_value = f"bytes={offset}-{offset+length-1}"
        resp = self._session.get(file_url, headers={
            "Range": range_value
        })
        resp.raise_for_status()
        content = resp.content
        if len(content) != length:
            raise ShardedIOError(f"Getting {file_url} error. Expecting "
                                 f"{length} bytes (Range: {range_value}), "
                                 f"but got {len(content)}.")
        return content

    def fetch_cmc_chunk(self, cmc: np.uint64):
        minishard_key = self.get_minishard_key(cmc)
        assert minishard_key in self.minishard_dict
        return self.minishard_dict[minishard_key].fetch_cmc_chunk(cmc)


class HttpShardedScale(ShardedScaleBase):

    can_read_cmc = True
    can_write_cmc = False

    def __init__(self, base_url: str, session: requests.Session, key: str,
                 shard_spec: ShardSpec,
                 shard_volume_spec: ShardVolumeSpec):
        super().__init__(key, shard_spec, shard_volume_spec)
        self.base_url = base_url
        self._session = session
        self.shard_dict: Dict[np.uint64, HttpShard] = {}

    def get_shard(self, shard_key: np.uint64) -> CMCReadWrite:
        if shard_key not in self.shard_dict:
            http_shard = HttpShard(f"{self.base_url}{self.key}/",
                                   self._session,
                                   shard_key,
                                   self.shard_spec)
            self.shard_dict[shard_key] = http_shard
        return self.shard_dict[shard_key]


class ShardedHttpAccessor(neuroglancer_scripts.http_accessor.HttpAccessor,
                          ShardedAccessorBase):
    def __init__(self, base_url):
        neuroglancer_scripts.http_accessor.HttpAccessor.__init__(self,
                                                                 base_url)
        self.shard_dict: Dict[str, HttpShardedScale] = {}
        self.info = json.loads(self.fetch_file("info"))

    def fetch_chunk(self, key, chunk_coords):
        if key not in self.shard_dict:
            sharding = self.get_sharding_spec(key)
            scale = self.get_scale(key)
            chunk_sizes, = scale.get("chunk_sizes", [[]])
            sizes = scale.get("size")
            shard_spec = ShardSpec(**sharding)
            shard_volume_spec = ShardVolumeSpec(chunk_sizes, sizes)
            self.shard_dict[key] = HttpShardedScale(self.base_url,
                                                    self._session,
                                                    key,
                                                    shard_spec,
                                                    shard_volume_spec)
        return self.shard_dict[key].fetch_chunk(chunk_coords)
