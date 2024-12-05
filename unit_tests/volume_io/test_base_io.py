from unittest.mock import PropertyMock, patch

import pytest
from neuroglancer_scripts.volume_io.base_io import MultiResIOBase


class DummyIO(MultiResIOBase):
    @property
    def info(self):
        pass

    def read_chunk(self, scale_key, chunk_coords):
        return super().read_chunk(scale_key, chunk_coords)

    def write_chunk(self, chunk, scale_key, chunk_coords):
        return super().write_chunk(chunk, scale_key, chunk_coords)


dummy_info = {
    "scales": [
        {"key": "scale0", "foo": "bar"},
        {"key": "scale1", "hello": "world"},
    ]
}


@pytest.fixture
def patched_dummy_io_info():
    with patch.object(DummyIO, "info", new_callable=PropertyMock) as info_mock:
        info_mock.return_value = dummy_info
        yield info_mock


def test_iter_scale(patched_dummy_io_info):
    io = DummyIO()
    assert [
        ("scale0", {"key": "scale0", "foo": "bar"}),
        ("scale1", {"key": "scale1", "hello": "world"}),
    ] == list(io.iter_scale())


@pytest.mark.parametrize(
    "key, error, expected_value",
    [
        ("scale0", None, {"key": "scale0", "foo": "bar"}),
        ("scale1", None, {"key": "scale1", "hello": "world"}),
        ("foo", IndexError, None),
    ],
)
def test_scale_info(key, error, expected_value, patched_dummy_io_info):
    io = DummyIO()
    if error:
        with pytest.raises(error):
            io.scale_info(key)
        return
    assert expected_value == io.scale_info(key)
