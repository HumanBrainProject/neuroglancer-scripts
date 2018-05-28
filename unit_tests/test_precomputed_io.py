# Copyright (c) 2018 CEA
# Author: Yann Leprince <yann.leprince@cea.fr>
#
# This software is made available under the MIT licence, see LICENCE.txt.

import numpy as np
import pytest

from neuroglancer_scripts.accessor import get_accessor_for_url
from neuroglancer_scripts.chunk_encoding import InvalidInfoError
from neuroglancer_scripts.precomputed_io import (
    get_IO_for_existing_dataset,
    get_IO_for_new_dataset,
)


DUMMY_INFO = {
    "type": "image",
    "data_type": "uint16",
    "num_channels": 1,
    "scales": [
        {
            "key": "key",
            "size": [8, 3, 15],
            "resolution": [1e6, 1e6, 1e6],
            "voxel_offset": [0, 0, 0],
            "chunk_sizes": [[8, 8, 8]],
            "encoding": "raw",
        }
    ]
}


def test_precomputed_IO_chunk_roundtrip(tmpdir):
    accessor = get_accessor_for_url(str(tmpdir))
    # Minimal info file
    io = get_IO_for_new_dataset(DUMMY_INFO, accessor)
    dummy_chunk = np.arange(8 * 3 * 7, dtype="uint16").reshape(1, 7, 3, 8)
    chunk_coords = (0, 8, 0, 3, 8, 15)
    io.write_chunk(dummy_chunk, "key", chunk_coords)
    assert np.array_equal(io.read_chunk("key", chunk_coords), dummy_chunk)

    io2 = get_IO_for_existing_dataset(accessor)
    assert io2.info == DUMMY_INFO
    assert np.array_equal(io2.read_chunk("key", chunk_coords), dummy_chunk)


def test_precomputed_IO_info_error(tmpdir):
    with (tmpdir / "info").open("w") as f:
        f.write("invalid JSON")
    accessor = get_accessor_for_url(str(tmpdir))
    with pytest.raises(InvalidInfoError):
        get_IO_for_existing_dataset(accessor)


def test_precomputed_IO_validate_chunk_coords(tmpdir):
    accessor = get_accessor_for_url(str(tmpdir))
    # Minimal info file
    io = get_IO_for_new_dataset(DUMMY_INFO, accessor)
    good_chunk_coords = (0, 8, 0, 3, 0, 8)
    bad_chunk_coords = (0, 8, 1, 4, 0, 8)
    assert io.validate_chunk_coords("key", good_chunk_coords) is True
    assert io.validate_chunk_coords("key", bad_chunk_coords) is False
