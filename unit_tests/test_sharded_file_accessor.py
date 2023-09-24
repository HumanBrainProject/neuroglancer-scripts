import pytest
from neuroglancer_scripts.sharded_file_accessor import (
    ShardedFileAccessor, ShardedScale
)
import numpy as np

default_kwargs = {
    "grid_sizes": [64, 64, 64],
    "chunk_sizes": [64, 64, 64],
}

@pytest.fixture
def sharded_scale():
    return ShardedScale(
        "foo",
        "key",
        **default_kwargs
    )

@pytest.mark.parametrize("chunk_coords,expected", [
    ((0, 64, 0, 64, 0, 64), 0x0),
    ((64, 128, 0, 64, 0, 64), 0x1),
    ((0, 64, 64, 128, 0, 64), 0x2),
    ((64, 128, 64, 128, 0, 64), 0x3),
    ((64, 128, 64, 128, 64, 128), 0x7),
    ((128, 196, 0, 64, 0, 64), 0x8),
])
def test_compressed_morton_code(sharded_scale,chunk_coords,expected):
    assert expected == sharded_scale.compressed_morton_code(chunk_coords), f"{chunk_coords!r},{expected!r}"


@pytest.mark.parametrize("shard_bits,expected_shard_mask", [
    (1, 0b1),
    (2, 0b11),
    (3, 0b111),
    (4, 0b1111),
])
def test_shard_mask(shard_bits,expected_shard_mask):
    sharded_scale = ShardedScale("foo", "key", shard_bits=shard_bits, **default_kwargs)
    assert sharded_scale.shard_mask == expected_shard_mask

@pytest.mark.parametrize("shard_bits,possible_keys", [
    (1, {f"{hex(i)[2:]}.shard" for i in range(2 ** 1)}),
    (2, {f"{hex(i)[2:]}.shard" for i in range(2 ** 2)}),
    (3, {f"{hex(i)[2:]}.shard" for i in range(2 ** 3)}),
    (4, {f"{hex(i)[2:]}.shard" for i in range(2 ** 4)}),
])
def test_get_shard_key(shard_bits,possible_keys):
    sharded_scale = ShardedScale("foo", shard_bits=shard_bits, **default_kwargs)
    assert {
        sharded_scale.get_shard_key(np.uint64(num))
        for num in range(1024)
    } == possible_keys
