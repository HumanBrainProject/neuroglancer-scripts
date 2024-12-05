import json
from itertools import chain, repeat
from unittest.mock import call, patch

import numpy as np
import pytest
from neuroglancer_scripts.accessor import Accessor, get_accessor_for_url
from neuroglancer_scripts.volume_io.n5_io import N5IO


@pytest.fixture
def mock_accessor_fetch_file():
    with patch.object(Accessor, "fetch_file") as fetch_file_mock:
        yield fetch_file_mock


no_scales_root_attr = {
    "downsamplingFactors": [],
    "dataType": "utf-16",
    "multiScale": True,
    "resolution": [1, 1, 1],
    "unit": ["m", "m", "m"],
}

sample_root_attr = {
    "downsamplingFactors": [
        [1, 1, 1],
        [2, 2, 2],
        [4, 4, 4],
    ],
    "dataType": "utf-16",
    "multiScale": True,
    "resolution": [1, 1, 1],
    "unit": ["m", "m", "m"],
}

sample_scale_attr = {
    "dataType": "uint16",
    "blockSize": [8, 8, 8],
    "dimensions": [1, 1, 1],
    "compression": {"type": "gzip"},
}
sample_scale_attr_bytes = json.dumps(sample_scale_attr)


@pytest.mark.parametrize(
    "fetch_file_sideeffects, error, expected_calls",
    [
        (repeat(b"foobar"), json.JSONDecodeError, [call("attributes.json")]),
        (
            chain(
                [json.dumps(no_scales_root_attr).encode("utf-8")],
                repeat(json.dumps(sample_scale_attr).encode("utf-8")),
            ),
            None,
            [call("attributes.json")],
        ),
        (
            chain(
                [json.dumps(sample_root_attr).encode("utf-8")],
                repeat(json.dumps(sample_scale_attr).encode("utf-8")),
            ),
            None,
            [
                call("attributes.json"),
                call("s0/attributes.json"),
                call("s1/attributes.json"),
                call("s2/attributes.json"),
            ],
        ),
    ],
)
def test_n5_init(
    fetch_file_sideeffects, error, expected_calls, mock_accessor_fetch_file
):
    mock_accessor_fetch_file.side_effect = fetch_file_sideeffects

    accessor = Accessor()
    accessor.can_read = True
    if error is not None:
        with pytest.raises(error):
            N5IO(accessor)
        assert mock_accessor_fetch_file.call_args_list == expected_calls
        return
    N5IO(accessor)
    assert mock_accessor_fetch_file.call_args_list == expected_calls


def test_n5io_roundtrip(tmpdir):
    accessor = get_accessor_for_url(str(tmpdir))

    root = {
        "downsamplingFactors": [[1, 1, 1]],
        "dataType": "utf-16",
        "multiScale": True,
        "resolution": [1, 1, 1],
        "unit": ["m", "m", "m"],
    }
    s0 = {
        "dataType": "uint16",
        "blockSize": [8, 3, 7],
        "dimensions": [1, 1, 1],
        "compression": {"type": "gzip"},
    }

    accessor.store_file("attributes.json", json.dumps(root).encode("utf-8"))
    accessor.store_file("s0/attributes.json", json.dumps(s0).encode("utf-8"))

    dummy_chunk = np.arange(8 * 3 * 7, dtype="uint16").reshape(1, 7, 3, 8)
    io = N5IO(accessor)
    io.write_chunk(dummy_chunk, "s0", (0, 8, 0, 3, 0, 7))

    retrieved_chunk = io.read_chunk("s0", (0, 8, 0, 3, 0, 7))
    assert np.array_equal(dummy_chunk, retrieved_chunk)
