import pytest
from collections import namedtuple
import numpy as np

from neuroglancer_scripts.sharded_base import (
    ShardVolumeSpec, ShardSpec, CMCReadWrite
)


ShardSpecArg = namedtuple("ShardSpecArg", ["minishard_bits", "shard_bits",
                                           "hash",
                                           "minishard_index_encoding",
                                           "data_encoding",
                                           "preshift_bits"])


ExpRes = namedtuple("ExpRes", "shard_mask minishard_mask preshift_mask error")
ExpRes.__new__.__defaults__ = (None,) * len(ExpRes._fields)

shard_spec_args = [
    (ShardSpecArg(1, 1, "identity", "raw", "raw", 0), ExpRes(0b10, 0b1)),

    (ShardSpecArg(2, 2, "identity", "gzip", "raw", 0), ExpRes(0b1100, 0b11)),
    (ShardSpecArg(2, 2, "identity", "raw", "gzip", 0), ExpRes(0b1100, 0b11)),
    (ShardSpecArg(2, 2, "identity", "gzip", "gzip", 0), ExpRes(0b1100, 0b11)),

    (ShardSpecArg(2, 2, "identity", "raw", "raw", -1), ExpRes(error=True)),
    (ShardSpecArg(-1, 2, "identity", "raw", "raw", 0), ExpRes(error=True)),
    (ShardSpecArg(2, -1, "identity", "raw", "raw", 0), ExpRes(error=True)),
    (
        ShardSpecArg(2, 2, "murmurhash3_x86_128", "raw", "raw", 0),
        ExpRes(error=True)
    ),
    (ShardSpecArg(2, 2, "identity", "zlib", "raw", 0), ExpRes(error=True)),
    (ShardSpecArg(2, 2, "identity", "raw", "foobar", 0), ExpRes(error=True)),
]


@pytest.mark.parametrize("shard_spec_arg, expected_result", shard_spec_args)
def test_shard_spec(shard_spec_arg: ShardSpecArg, expected_result: ExpRes):
    if expected_result.error:

        with pytest.raises(Exception):
            ShardSpec(*shard_spec_arg)
        return

    shard_spec = ShardSpec(*shard_spec_arg)
    print("shard_spec.minishard_mask", shard_spec.minishard_mask)
    assert shard_spec.minishard_mask == expected_result.minishard_mask
    assert shard_spec.shard_mask == expected_result.shard_mask


@pytest.fixture
def shard_volume_spec():
    return ShardVolumeSpec(
        [64, 64, 64],
        [6400, 6400, 6400]
    )


@pytest.mark.parametrize("chunk_coords,expected", [
    ((0, 64, 0, 64, 0, 64), 0x0),
    ((64, 128, 0, 64, 0, 64), 0x1),
    ((0, 64, 64, 128, 0, 64), 0x2),
    ((64, 128, 64, 128, 0, 64), 0x3),
    ((64, 128, 64, 128, 64, 128), 0x7),
    ((128, 196, 0, 64, 0, 64), 0x8),
])
def test_compressed_morton_code(shard_volume_spec: ShardVolumeSpec,
                                chunk_coords,
                                expected):
    cmc = shard_volume_spec.get_cmc(chunk_coords)
    assert cmc == expected


class Foo(CMCReadWrite):
    pass


shard_key_args = range(1, 8)


@pytest.mark.parametrize("shard_bits", shard_key_args)
def test_get_shard_key(shard_bits):
    shard_spec = ShardSpec(minishard_bits=0, shard_bits=shard_bits)
    expected_keys = {
        np.uint64(i)
        for i in range(2 ** int(shard_bits))
    }
    cmc_rw = Foo(shard_spec)
    assert {
        cmc_rw.get_shard_key(np.uint64(num))
        for num in range(1024)
    } == expected_keys
