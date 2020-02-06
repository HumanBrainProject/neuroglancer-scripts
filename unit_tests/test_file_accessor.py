# Copyright (c) 2018 Forschungszentrum Juelich GmbH
# Author: Yann Leprince <y.leprince@fz-juelich.de>
#
# This software is made available under the MIT licence, see LICENCE.txt.

import pathlib

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


def test_file_accessor_file_exists(tmpdir):
    a = FileAccessor(str(tmpdir))
    assert a.file_exists("nonexistent_file") is False
    (tmpdir / "real_file").open("w")  # create an empty file
    assert a.file_exists("real_file") is True
    assert a.file_exists("nonexistent_dir/file") is False


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


def test_file_accessor_errors(tmpdir):
    # tmpdir from pytest is missing features of pathlib
    tmpdir = pathlib.Path(str(tmpdir))
    a = FileAccessor(str(tmpdir))
    chunk_coords = (0, 1, 0, 1, 0, 1)
    with pytest.raises(DataAccessError):
        a.fetch_file("info")
    with pytest.raises(DataAccessError):
        a.fetch_chunk("key", chunk_coords)

    inaccessible_file = tmpdir / "inaccessible"
    inaccessible_file.touch(mode=0o000, exist_ok=False)
    with pytest.raises(DataAccessError):
        a.fetch_file("inaccessible")

    inaccessible_chunk = tmpdir / "inaccessible_key" / "0-1_0-1_0-1"
    inaccessible_chunk.parent.mkdir(mode=0o000)
    with pytest.raises(DataAccessError):
        a.fetch_chunk("inaccessible_key", chunk_coords)
    with pytest.raises(DataAccessError):
        a.store_chunk(b"", "inaccessible_key", chunk_coords)

    with pytest.raises(DataAccessError):
        a.file_exists("inaccessible_key/dummy")
    with pytest.raises(DataAccessError):
        a.store_file("inaccessible_key/dummy", b"")

    # Allow pytest to remove tmpdir with os.rmtree
    inaccessible_chunk.parent.chmod(mode=0o755)

    invalid_gzip_file = tmpdir / "invalid.gz"
    with invalid_gzip_file.open("w") as f:
        f.write("not gzip compressed")
    with pytest.raises(DataAccessError):
        print(a.fetch_file("invalid"))

    a.store_file("existing", b"")
    with pytest.raises(DataAccessError):
        a.store_file("existing", b"", overwrite=False)
    a.store_file("existing", b"", overwrite=True)

    with pytest.raises(ValueError):
        a.file_exists("../forbidden")
    with pytest.raises(ValueError):
        a.fetch_file("../forbidden")
    with pytest.raises(ValueError):
        a.store_file("../forbidden", b"")
