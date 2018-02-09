# Copyright (c) 2018 Forschungszentrum Juelich GmbH
# Author: Yann Leprince <y.leprince@fz-juelich.de>
#
# This software is made available under the MIT licence, see LICENCE.txt.

import numpy as np
import pytest

from neuroglancer_scripts.file_accessor import *
from neuroglancer_scripts.accessor import *


@pytest.mark.parametrize("flat", [False, True])
@pytest.mark.parametrize("gzip", [False, True])
def test_file_accessor_roundtrip(tmpdir, gzip, flat):
    a = FileAccessor(str(tmpdir), gzip=gzip, flat=flat)
    fake_info = {"scales": [{"key": "key"}]}
    fake_chunk_buf = b"d a t a"
    chunk_coords = (0, 1, 0, 1, 0, 1)
    a.store_info(fake_info)
    assert a.fetch_info() == fake_info
    a.store_chunk(fake_chunk_buf, "key", chunk_coords,
                  already_compressed=False)
    assert a.fetch_chunk("key", chunk_coords) == fake_chunk_buf
    chunk_coords2 = (0, 1, 0, 1, 1, 2)
    a.store_chunk(fake_chunk_buf, "key", chunk_coords2,
                  already_compressed=True)
    assert a.fetch_chunk("key", chunk_coords2) == fake_chunk_buf


def test_file_accessor_nonexistent_directory():
    a = FileAccessor("/nonexistent/directory")
    with pytest.raises(DataAccessError):
        a.fetch_info()
    with pytest.raises(DataAccessError):
        a.store_info({})
    chunk_coords = (0, 1, 0, 1, 0, 1)
    with pytest.raises(DataAccessError):
        a.fetch_chunk("key", chunk_coords)
    with pytest.raises(DataAccessError):
        a.store_chunk(b"", "key", chunk_coords)


def test_file_accessor_invalid_fetch(tmpdir):
    a = FileAccessor(str(tmpdir))
    chunk_coords = (0, 1, 0, 1, 0, 1)
    with pytest.raises(DataAccessError):
        a.fetch_info()
    with pytest.raises(DataAccessError):
        a.fetch_chunk("key", (0, 1, 0, 1, 0, 1))
