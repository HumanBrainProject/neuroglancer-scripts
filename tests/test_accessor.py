# Copyright (c) 2018 Forschungszentrum Juelich GmbH
# Author: Yann Leprince <y.leprince@fz-juelich.de>
#
# This software is made available under the MIT licence, see LICENCE.txt.

import pytest

from neuroglancer_scripts.accessor import *
from neuroglancer_scripts.file_accessor import FileAccessor
from neuroglancer_scripts.http_accessor import HttpAccessor


@pytest.mark.parametrize("accessor_options", [
    {},
    {"gzip": True, "flat": False, "unknown_option": None},
])
def test_get_accessor_for_url(accessor_options):
    assert isinstance(get_accessor_for_url(""), Accessor)
    a = get_accessor_for_url(".", accessor_options)
    assert isinstance(a, FileAccessor)
    assert a.base_dir == "."
    a = get_accessor_for_url("file:///absolute", accessor_options)
    assert isinstance(a, FileAccessor)
    assert a.base_dir == "/absolute"
    a = get_accessor_for_url("http://example/", accessor_options)
    assert isinstance(a, HttpAccessor)
    assert a.base_url == "http://example/"


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
    assert convert_file_url_to_pathname("file:///with%20space") == "/with space"
    assert convert_file_url_to_pathname("precomputed://file:///") == "/"
