# Copyright (c) 2018 CEA
# Author: Yann Leprince <yann.leprince@cea.fr>
#
# This software is made available under the MIT licence, see LICENCE.txt.

import json

import pytest
import requests

from neuroglancer_scripts.http_accessor import HttpAccessor
from neuroglancer_scripts.accessor import (
    DataAccessError,
)


@pytest.mark.parametrize("base_url", [
    "http://h.test/i/",
    "http://h.test/i",
])
def test_http_accessor(base_url, requests_mock):
    dummy_info = {"scales": [{"key": "key"}]}
    dummy_chunk_buf = b"d a t a"
    chunk_coords = (0, 1, 0, 1, 0, 1)
    a = HttpAccessor(base_url)

    requests_mock.get("http://h.test/i/info", json=dummy_info)
    fetched_info = a.fetch_file("info")
    assert json.loads(fetched_info.decode()) == dummy_info

    requests_mock.head("http://h.test/i/info", status_code=200)
    assert a.file_exists("info") is True

    requests_mock.head("http://h.test/i/info", status_code=404)
    assert a.file_exists("info") is False

    requests_mock.get("http://h.test/i/key/0-1_0-1_0-1",
                      content=dummy_chunk_buf)
    fetched_chunk = a.fetch_chunk("key", chunk_coords)
    assert fetched_chunk == dummy_chunk_buf


def test_http_accessor_errors(requests_mock):
    chunk_coords = (0, 1, 0, 1, 0, 1)
    a = HttpAccessor("http://h.test/i/")

    requests_mock.head("http://h.test/i/info", status_code=500)
    with pytest.raises(DataAccessError):
        a.file_exists("info")

    requests_mock.get("http://h.test/i/info", status_code=404)
    with pytest.raises(DataAccessError):
        a.fetch_file("info")

    requests_mock.get("http://h.test/i/info",
                      exc=requests.exceptions.ConnectTimeout)
    with pytest.raises(DataAccessError):
        a.fetch_file("info")

    requests_mock.get("http://h.test/i/key/0-1_0-1_0-1", status_code=404)
    with pytest.raises(DataAccessError):
        a.fetch_chunk("key", chunk_coords)
