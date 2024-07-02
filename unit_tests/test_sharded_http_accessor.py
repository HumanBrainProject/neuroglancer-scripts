# Copyright (c) 2023, 2024 Forschungszentrum Juelich GmbH
# Author: Xiao Gui <xgui3783@gmail.com>
#
# This software is made available under the MIT licence, see LICENCE.txt.

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from neuroglancer_scripts.sharded_http_accessor import (
    HttpShard,
    HttpShardedScale,
    ShardedHttpAccessor,
    ShardSpec,
    ShardVolumeSpec,
)
from requests import Session
from requests import exceptions as re_exc

hdr_len = int(2 ** 2 * 16)


@pytest.fixture
def shard_spec():
    return ShardSpec(2, 2)


@pytest.fixture
def shard_key():
    return np.uint64(1), "1"


# HttpShard
@pytest.mark.parametrize("base_url", [
    "http://test-foo/path/20um/",
    "http://test-foo/path/20um"
])
@pytest.mark.parametrize("get_mock_result, exp_is_legacy, err_flag", [
    ((True, True, False), True, False),
    ((False, False, True), False, False),
    ((True, False, True), False, False),
    ((False, True, True), False, False),
    ((True, True, True), False, False),

    ((False, False, False), None, True),
])
@patch.object(HttpShard, "get_minishards_offsets", return_value=[])
def test_http_shard(get_msh_offset_m, base_url, shard_key, shard_spec,
                    requests_mock, get_mock_result, exp_is_legacy, err_flag):

    base_url = base_url if base_url.endswith("/") else (base_url + "/")
    shard_key, shard_key_str = shard_key

    for mresult, ext in zip(get_mock_result, ["index", "data", "shard"]):
        status = 200 if mresult else 404
        requests_mock.head(f"{base_url}{shard_key_str}.{ext}",
                           status_code=status)

    if err_flag:
        with pytest.raises(Exception):
            HttpShard(base_url, Session(), shard_key, shard_spec)
            pass
        get_msh_offset_m.assert_not_called()
        return

    shard = HttpShard(base_url, Session(), shard_key, shard_spec)
    assert shard.is_legacy == exp_is_legacy


sc_base_url = "http://test-foo/path/20um/"


@pytest.fixture
def legacy_http_shard(shard_key, shard_spec, requests_mock):
    shard_key, shard_key_str = shard_key
    requests_mock.head(f"{sc_base_url}{shard_key_str}.index", status_code=200)
    requests_mock.head(f"{sc_base_url}{shard_key_str}.data", status_code=200)
    requests_mock.head(f"{sc_base_url}{shard_key_str}.shard", status_code=404)
    with patch.object(HttpShard, "get_minishards_offsets", return_value=[]):
        yield HttpShard(sc_base_url, Session(), shard_key, shard_spec)


@pytest.fixture
def modern_http_shard(shard_key, shard_spec, requests_mock):
    shard_key, shard_key_str = shard_key
    requests_mock.head(f"{sc_base_url}{shard_key_str}.index", status_code=404)
    requests_mock.head(f"{sc_base_url}{shard_key_str}.data", status_code=404)
    requests_mock.head(f"{sc_base_url}{shard_key_str}.shard", status_code=200)
    with patch.object(HttpShard, "get_minishards_offsets", return_value=[]):
        yield HttpShard(sc_base_url, Session(), shard_key, shard_spec)


@pytest.mark.parametrize("f_name", [
    "legacy_http_shard",
    "modern_http_shard",
])
def test_sharded_http_file_exists(f_name, requests_mock,
                                  request):
    shard: HttpShard = request.getfixturevalue(f_name)
    exists = "exist.txt"
    notexists = "notexists.txt"
    error = "error.txt"
    nerror = "networkerr.txt"
    requests_mock.head(f"{sc_base_url}{exists}", status_code=200)
    requests_mock.head(f"{sc_base_url}{notexists}", status_code=404)
    requests_mock.head(f"{sc_base_url}{error}", status_code=500)
    requests_mock.head(f"{sc_base_url}{nerror}", exc=re_exc.ConnectTimeout)
    assert shard.file_exists(exists)
    assert not shard.file_exists(notexists)
    with pytest.raises(Exception):
        shard.file_exists(error)
    with pytest.raises(Exception):
        shard.file_exists(nerror)


@pytest.mark.parametrize("offsetlen", [
    (0, hdr_len),
    (hdr_len, 5),
])
@pytest.mark.parametrize("f_name", [
    "legacy_http_shard",
    "modern_http_shard",
])
def test_sharded_http_read_bytes(f_name, requests_mock, offsetlen,
                                 request):
    shard: HttpShard = request.getfixturevalue(f_name)
    offset, length = offsetlen

    filename = "1.shard"
    if f_name == "legacy_http_shard":
        filename = "1.index" if (offset < hdr_len) else "1.data"
        offset = offset if offset < hdr_len else (offset - hdr_len)

    requests_mock.get(f"{sc_base_url}{filename}", request_headers={
        "Range": f"bytes={offset}-{offset + length - 1}"
    }, content=b"\0" * length)

    assert b"\0" * length == shard.read_bytes(*offsetlen)

    with pytest.raises(Exception):
        requests_mock.get(f"{sc_base_url}{filename}", request_headers={
            "Range": f"bytes={offset}-{offset + length - 1}"
        }, content=b"\0" * (length + 1))
        shard.read_bytes(*offsetlen)


def test_sharded_http_fetch_cmc(modern_http_shard):
    minishard_key = "foo-bar"
    cmc = np.uint64(123)
    with patch.object(modern_http_shard, "get_minishard_key",
                      return_value=minishard_key):
        assert len(modern_http_shard.minishard_dict) == 0
        with pytest.raises(Exception):
            modern_http_shard.fetch_cmc_chunk(cmc)

        mock_minishard = MagicMock()
        modern_http_shard.minishard_dict[minishard_key] = mock_minishard
        mock_minishard.fetch_cmc_chunk.return_value = b"foo-bar"

        assert modern_http_shard.fetch_cmc_chunk(cmc) == b"foo-bar"
        mock_minishard.fetch_cmc_chunk.assert_called_once_with(cmc)


base_url = "http://test-foo/path"


@pytest.fixture
def shard_volume_spec():
    return ShardVolumeSpec([64, 64, 64], [128, 128, 128])


class DummyCls:
    def __init__(self, *args, **kwargs):
        ...


# HttpShardedScale
@pytest.mark.parametrize("key_exists", [True, False])
@patch("neuroglancer_scripts.sharded_http_accessor.HttpShard", DummyCls)
def test_http_sharded_scale_get_shard(key_exists, shard_spec,
                                      shard_volume_spec):
    scale = HttpShardedScale(base_url, Session(), "20um", shard_spec,
                             shard_volume_spec)

    scale.shard_dict = MagicMock()
    scale.shard_dict.__contains__.return_value = key_exists
    scale.shard_dict.__getitem__.return_value = "foo-bar"

    cmc = np.uint64(123)
    return_val = scale.get_shard(cmc)

    scale.shard_dict.__contains__.assert_called_once_with(cmc)
    if key_exists:
        scale.shard_dict.__setitem__.assert_not_called()
    else:
        scale.shard_dict.__setitem__.assert_called_once()
    scale.shard_dict.__getitem__.assert_called_once_with(cmc)
    assert return_val == "foo-bar"


@pytest.fixture
def sh_http_accessor(requests_mock):
    requests_mock.get(f"{base_url}/info", json={
        "scales": [
            {
                "key": "20mm",
                "size": [256, 256, 256],
                "chunk_sizes": [
                    [64, 64, 64]
                ],
                "sharding": {
                    "@type": "neuroglancer_uint64_sharded_v1",
                    "shard_bits": 2,
                    "minishard_bits": 2
                }
            },
            {
                "key": "40mm",
                "size": [128, 128, 128],
                "chunk_sizes": [
                    [64, 64, 64]
                ],
                "sharding": {
                    "@type": "neuroglancer_uint64_sharded_v1",
                    "shard_bits": 2,
                    "minishard_bits": 2
                }
            }
        ]
    })
    requests_mock.get(f"{base_url}/20mm/0.shard", status_code=200)
    return ShardedHttpAccessor(base_url)


@pytest.mark.parametrize("key_exists", [True, False])
def test_sharded_http_accessor(sh_http_accessor, key_exists):

    sh_sc_dict = sh_http_accessor.shard_scale_dict = MagicMock()
    sh_sc_dict.__contains__.return_value = key_exists
    mocked_sharded_scale = sh_sc_dict.__getitem__.return_value
    mocked_sharded_scale.fetch_chunk.return_value = b"foo-bar"

    coord = (64, 128, 0, None, 0, ["foo-bar"])
    return_val = sh_http_accessor.fetch_chunk("20mm", coord)

    sh_sc_dict.__contains__.assert_called_once_with("20mm")
    if key_exists:
        sh_sc_dict.__setitem__.assert_not_called()
    else:
        sh_sc_dict.__setitem__.assert_called_once()
    sh_sc_dict.__getitem__.assert_called_once_with("20mm")
    assert return_val == b"foo-bar"
