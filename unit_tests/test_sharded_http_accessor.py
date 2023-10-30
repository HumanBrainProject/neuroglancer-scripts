import pytest

from requests import Session

from neuroglancer_scripts.sharded_http_accessor import (
    HttpShard,
    ShardSpec
)


@pytest.fixture
def session():
    return Session()


@pytest.fixture
def shard_spec():
    return ShardSpec(2, 2)


# HttpShard
@pytest.fixture
def http_shard(shard_spec, session):
    return HttpShard()
