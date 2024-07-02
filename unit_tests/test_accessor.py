# Copyright (c) 2018, 2023 Forschungszentrum Juelich GmbH
# Author: Yann Leprince <y.leprince@fz-juelich.de>
# Author: Xiao Gui <xgui3783@gmail.com>
#
# This software is made available under the MIT licence, see LICENCE.txt.

import argparse
import json
import pathlib
from unittest.mock import patch

import pytest
from neuroglancer_scripts.accessor import (
    Accessor,
    DataAccessError,
    URLError,
    add_argparse_options,
    convert_file_url_to_pathname,
    get_accessor_for_url,
)
from neuroglancer_scripts.file_accessor import FileAccessor
from neuroglancer_scripts.http_accessor import HttpAccessor
from neuroglancer_scripts.sharded_base import ShardedAccessorBase
from neuroglancer_scripts.sharded_file_accessor import ShardedFileAccessor
from neuroglancer_scripts.sharded_http_accessor import ShardedHttpAccessor


@pytest.mark.parametrize("accessor_options", [
    {},
    {"gzip": True, "flat": False, "unknown_option": None},
])
def test_get_accessor_for_url(accessor_options):
    assert isinstance(get_accessor_for_url(""), Accessor)
    a = get_accessor_for_url(".", accessor_options)
    assert isinstance(a, FileAccessor)
    assert a.base_path == pathlib.Path(".")
    a = get_accessor_for_url("file:///absolute", accessor_options)
    assert isinstance(a, FileAccessor)
    assert a.base_path == pathlib.Path("/absolute")
    a = get_accessor_for_url("http://example/", accessor_options)
    assert isinstance(a, HttpAccessor)
    assert a.base_url == "http://example/"
    with pytest.raises(URLError, match="scheme"):
        get_accessor_for_url("weird://", accessor_options)
    with pytest.raises(URLError, match="decod"):
        get_accessor_for_url("file:///%ff", accessor_options)


valid_info_str = json.dumps({
            "scales": [
                {
                    "key": "foo",
                    "sharding": {
                        "@type": "neuroglancer_uint64_sharded_v1"
                    }
                }
            ]
        })


@patch.object(ShardedAccessorBase, "info_is_sharded")
@pytest.mark.parametrize("scheme", ["https://", "http://", ""])
@pytest.mark.parametrize("fetch_file_returns, info_is_sharded_returns, exp", [
    (DataAccessError("foobar"), None, False),
    ('mal formed json', None, False),
    (valid_info_str, False, False),
    (valid_info_str, True, True),
])
def test_sharded_accessor_via_info(info_is_sharded_mock, fetch_file_returns,
                                   info_is_sharded_returns, exp, scheme):

    if isinstance(info_is_sharded_returns, Exception):
        info_is_sharded_mock.side_effect = info_is_sharded_returns
    else:
        info_is_sharded_mock.return_value = info_is_sharded_returns

    assert scheme in ("https://", "http://", "file://", "")
    if scheme in ("file://", ""):
        base_acc_cls = FileAccessor
        shard_accessor_cls = ShardedFileAccessor
    if scheme in ("https://", "http://"):
        base_acc_cls = HttpAccessor
        shard_accessor_cls = ShardedHttpAccessor
    with patch.object(base_acc_cls, "fetch_file") as fetch_file_mock:
        if isinstance(fetch_file_returns, Exception):
            fetch_file_mock.side_effect = fetch_file_returns
        else:
            fetch_file_mock.return_value = fetch_file_returns

        result = get_accessor_for_url(f"{scheme}example/")
        assert isinstance(result, shard_accessor_cls if exp else base_acc_cls)

        if info_is_sharded_returns is None:
            info_is_sharded_mock.assert_not_called()
        else:
            info_is_sharded_mock.assert_called_once()


@pytest.mark.parametrize("write_chunks", [True, False])
@pytest.mark.parametrize("write_files", [True, False])
def test_add_argparse_options(write_chunks, write_files):
    # Test default values
    parser = argparse.ArgumentParser()
    add_argparse_options(parser,
                         write_chunks=write_chunks,
                         write_files=write_files)
    args = parser.parse_args([])
    get_accessor_for_url(".", vars(args))


def test_add_argparse_options_parsing():
    # Test correct parsing
    parser = argparse.ArgumentParser()
    add_argparse_options(parser)
    args = parser.parse_args(["--flat"])
    assert args.flat is True
    args = parser.parse_args(["--no-gzip"])
    assert args.gzip is False


def test_convert_file_url_to_pathname():
    assert convert_file_url_to_pathname("") == ""
    assert convert_file_url_to_pathname("relative/path") == "relative/path"
    assert (convert_file_url_to_pathname("relative/../path")
            == "relative/../path")
    assert (convert_file_url_to_pathname("/path/with spaces")
            == "/path/with spaces")
    assert convert_file_url_to_pathname("/absolute/path") == "/absolute/path"
    assert convert_file_url_to_pathname("file:///") == "/"
    with pytest.raises(URLError):
        convert_file_url_to_pathname("http://")
    with pytest.raises(URLError):
        convert_file_url_to_pathname("file://invalid/")
    assert convert_file_url_to_pathname("file:///test") == "/test"
    assert convert_file_url_to_pathname("file://localhost/test") == "/test"
    assert (convert_file_url_to_pathname("file:///with%20space")
            == "/with space")
    assert convert_file_url_to_pathname("precomputed://file:///") == "/"
