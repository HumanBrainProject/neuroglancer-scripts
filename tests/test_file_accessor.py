# Copyright (c) 2018 Forschungszentrum Juelich GmbH
# Author: Yann Leprince <y.leprince@fz-juelich.de>
#
# This software is made available under the MIT licence, see LICENCE.txt.

import pytest

from neuroglancer_scripts.file_accessor import FileAccessor
from neuroglancer_scripts.accessor import (
    DataAccessError,
)


@pytest.mark.parametrize("flat", [False, True])
@pytest.mark.parametrize("gzip", [False, True])
def test_file_accessor_roundtrip(tmpdir, gzip, flat):
    a = FileAccessor(str(tmpdir), gzip=gzip, flat=flat)
    fake_info = b'{"scales": [{"key": "key"}]}'
    fake_chunk_buf = b"d a t a"
    chunk_coords = (0, 1, 0, 1, 0, 1)
    a.store_file("info", fake_info, mime_type="application/json")
    assert a.fetch_file("info") == fake_info
    a.store_chunk(fake_chunk_buf, "key", chunk_coords,
                  mime_type="application/octet-stream")
    assert a.fetch_chunk("key", chunk_coords) == fake_chunk_buf
    chunk_coords2 = (0, 1, 0, 1, 1, 2)
    a.store_chunk(fake_chunk_buf, "key", chunk_coords2,
                  mime_type="image/jpeg")
    assert a.fetch_chunk("key", chunk_coords2) == fake_chunk_buf


def test_file_accessor_nonexistent_directory():
    a = FileAccessor("/nonexistent/directory")
    with pytest.raises(DataAccessError):
        a.fetch_file("info")
    with pytest.raises(DataAccessError):
        a.store_file("info", b"")
    chunk_coords = (0, 1, 0, 1, 0, 1)
    with pytest.raises(DataAccessError):
        a.fetch_chunk("key", chunk_coords)
    with pytest.raises(DataAccessError):
        a.store_chunk(b"", "key", chunk_coords)


def test_file_accessor_invalid_fetch(tmpdir):
    a = FileAccessor(str(tmpdir))
    chunk_coords = (0, 1, 0, 1, 0, 1)
    with pytest.raises(DataAccessError):
        a.fetch_file("info")
    with pytest.raises(DataAccessError):
        a.fetch_chunk("key", chunk_coords)
