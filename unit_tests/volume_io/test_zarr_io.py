import json
from unittest.mock import MagicMock, PropertyMock, call, patch

import numpy as np
import pytest
from neuroglancer_scripts.accessor import Accessor, get_accessor_for_url
from neuroglancer_scripts.volume_io.zarr_io import ChunkEncoder, ZarrV2IO


class MockAccessor(Accessor):
    pass


@pytest.mark.parametrize(
    "zgroup_content, zattrs_content, error",
    [
        (1, b'{"multiscales": [{}]}', TypeError),
        (b"<html></html>", b'{"multiscales": [{}]}', json.JSONDecodeError),
        (b'{"zarr_format": 2}', 1, TypeError),
        (b'{"zarr_format": 2}', b"<html></html>", json.JSONDecodeError),
        (b'{"zarr_format": 2, "foo": "bar"}',
         b'{"multiscales": [{}]}',
         AssertionError),
        (b'{"foo": "bar"}', b'{"multiscales": [{}]}', AssertionError),
        (b'{"zarr_format": 2}', b'{"foo": "bar"}', AssertionError),
        (b'{"zarr_format": 2}', b'{"multiscales": [{}]}', None),
    ],
)
def test_zarrv2_init(zgroup_content, zattrs_content, error):

    called_count = 0

    def mock_fetch(_, relative_path):
        nonlocal called_count
        called_count += 1
        if relative_path == ".zgroup":
            return zgroup_content
        if relative_path == ".zattrs":
            return zattrs_content
        raise NotImplementedError

    with patch.object(MockAccessor, "fetch_file", new=mock_fetch):
        acc = MockAccessor()
        if error is not None:
            with pytest.raises(error):
                ZarrV2IO(acc)
            return

        ZarrV2IO(acc)
        assert called_count == 2


@pytest.fixture
def mock_pass_verify():
    with patch.object(ZarrV2IO, "_verify", return_value=None) as verify_call:
        yield verify_call


@pytest.fixture
def mocked_multiscale_property(mock_pass_verify):
    with patch.object(
        ZarrV2IO, "multiscale", new_callable=PropertyMock,
    ) as multiscale_prop:
        yield multiscale_prop


@pytest.mark.parametrize(
    "multiscale_prop, error",
    [
        (None, AttributeError),
        ({"datasets": None}, TypeError),
        ({"datasets": [{"path": "1"}, {"foo": "bar"}]}, AssertionError),
        ({"datasets": [{"path": "1"}, {"path": "2"}]}, None),
        ({"datasets": []}, None),
        ({}, None),
    ],
)
def test_zarray_info_dict_vary_multiscale(
    multiscale_prop, error, mocked_multiscale_property,
):

    def mock_fetch(*args, **kwargs):
        return b'{"foo": "bar"}'

    with patch.object(MockAccessor, "fetch_file", new=mock_fetch):
        acc = MockAccessor()

        mocked_multiscale_property.return_value = multiscale_prop
        if error is not None:
            with pytest.raises(error):
                io = ZarrV2IO(acc)
                io.zarray_info_dict
            return
        io = ZarrV2IO(acc)
        assert io.zarray_info_dict is not None


class CustomError(Exception):
    pass


@pytest.mark.parametrize(
    "fetch_file_return_values, error",
    [
        ([CustomError, b'{"foo": "bar"}'], CustomError),
        ([b'{"foo": "bar"}', CustomError], CustomError),
        ([b"foobar", b'{"foo": "bar"}'], json.JSONDecodeError),
        ([b'{"foo": "bar"}', b"foobar"], json.JSONDecodeError),
        ([b'{"foo": "bar"}', b'{"foo": "bar"}'], None),
    ],
)
def test_zarray_info_dict_vary_fetch_file(
    fetch_file_return_values, error, mocked_multiscale_property,
):

    mocked_multiscale_property.return_value = {
        "datasets": [
            {"path": "1"},
            {"path": "2"},
        ],
    }
    counter = 0

    def mock_fetch(*args, **kwargs):
        nonlocal counter
        return_value = fetch_file_return_values[
            counter % len(fetch_file_return_values)]
        counter += 1
        if (
            isinstance(return_value, type)
            and issubclass(return_value, Exception)
        ):
            raise return_value
        return return_value

    with patch.object(MockAccessor, "fetch_file", new=mock_fetch):
        acc = MockAccessor()
        if error is not None:
            with pytest.raises(error):
                io = ZarrV2IO(acc)
                io.zarray_info_dict
            return

        io = ZarrV2IO(acc)
        assert io.zarray_info_dict is not None


def test_zarray_info_dict_valid(mocked_multiscale_property):

    mocked_multiscale_property.return_value = {
        "datasets": [
            {"path": "1"},
            {"path": "2"},
        ],
    }

    mock_accessor = MagicMock()
    mock_accessor.fetch_file.side_effect = [
        b'{"foo": "bar0"}',
        b'{"foo": "bar1"}']

    io = ZarrV2IO(mock_accessor)
    assert io.zarray_info_dict == {
        "1": {"foo": "bar0"},
        "2": {"foo": "bar1"},
    }

    assert mock_accessor.fetch_file.call_args_list == [
        call("1/.zarray"),
        call("2/.zarray"),
    ]


@pytest.fixture
def mock_get_encoder():
    with patch.object(ChunkEncoder, "get_encoder") as mock_get_encoder:
        yield mock_get_encoder


SCALE_KEY = "foo"


# TODO technically, dtype must be <u8 etc (i.e. byteorder + dtype)
@pytest.mark.parametrize(
    "zarray_info, scale_key, error, expected_compressor, expected_dtype, "
    "expected_numchanel",
    [
        (
            {SCALE_KEY: {"compressor": {"id": "gzip"}, "dtype": "uint8"}},
            f"{SCALE_KEY}foo",
            KeyError,
            None,
            None,
            None,
        ),
        (
            {SCALE_KEY: {"compressor": {"id": "gzip"}, "dtype": "uint8"}},
            SCALE_KEY,
            None,
            "gzip",
            np.uint8,
            1,
        ),
        (
            {SCALE_KEY: {"dtype": "uint8"}},
            SCALE_KEY,
            None,
            None,
            np.uint8,
            1,
        ),
        (
            {SCALE_KEY: {"compressor": {"id": "gzip"}}},
            SCALE_KEY,
            AssertionError,
            None,
            None,
            None,
        ),
    ],
)
def test_get_encoder(
    zarray_info,
    scale_key,
    error,
    expected_compressor,
    expected_dtype,
    expected_numchanel,
    mock_get_encoder,
    mock_pass_verify,
):
    with patch.object(
        ZarrV2IO, "zarray_info_dict", new_callable=PropertyMock,
    ) as zarray_info_dict_mock:
        zarray_info_dict_mock.return_value = zarray_info
        if error is not None:
            with pytest.raises(error):
                io = ZarrV2IO(None)
                io.get_encoder(scale_key)
            return
        io = ZarrV2IO(None)
        io.get_encoder(scale_key)
        mock_get_encoder.assert_called_with(
            expected_compressor, expected_dtype, expected_numchanel,
        )


@pytest.mark.parametrize(
    "zarray_info, scale_key, chunk_coords, error, expected_return",
    [
        (
            {SCALE_KEY: {"dimension_separator": "/", "chunks": [64, 64, 64]}},
            f"{SCALE_KEY}foo",
            (0, 64, 0, 64, 0, 64),
            KeyError,
            None,
        ),
        (
            {SCALE_KEY: {"chunks": [64, 64, 64]}},
            SCALE_KEY,
            (0, 64, 0, 64, 0, 64),
            None,
            f"{SCALE_KEY}/0.0.0",
        ),
        (
            {SCALE_KEY: {"dimension_separator": "/"}},
            SCALE_KEY,
            (0, 64, 0, 64, 0, 64),
            AssertionError,
            None,
        ),
        (
            {SCALE_KEY: {"dimension_separator": "_", "chunks": [64, 64, 64]}},
            SCALE_KEY,
            (0, 64, 0, 64, 0, 64),
            None,
            f"{SCALE_KEY}/0_0_0",
        ),
        (
            {SCALE_KEY: {"dimension_separator": "/", "chunks": [64, 64, 64]}},
            SCALE_KEY,
            (0, 64, 0, 64, 0, 64),
            None,
            f"{SCALE_KEY}/0/0/0",
        ),
        (
            {SCALE_KEY: {"dimension_separator": "/", "chunks": [64, 64, 64]}},
            SCALE_KEY,
            (0, 64, 64, 128, 0, 64),
            None,
            f"{SCALE_KEY}/0/1/0",
        ),
        (
            {SCALE_KEY: {"dimension_separator": "/", "chunks": [64, 64, 64]}},
            SCALE_KEY,
            (1, "HELLO WORLD", 64, 128, 150, None),
            None,
            f"{SCALE_KEY}/0/1/2",
        ),
        (
            {SCALE_KEY: {"dimension_separator": "/", "chunks": [64, 64, 64]}},
            SCALE_KEY,
            (-64, 0, 0, 64, 0, 64),
            AssertionError,
            None,
        ),
    ],
)
def test_format_path(
    zarray_info, scale_key, chunk_coords, error, expected_return,
    mock_pass_verify,
):

    with patch.object(
        ZarrV2IO, "zarray_info_dict", new_callable=PropertyMock,
    ) as zarray_info_dict_mock:
        zarray_info_dict_mock.return_value = zarray_info
        if error is not None:
            with pytest.raises(error):
                io = ZarrV2IO(None)
                io.format_path(scale_key, chunk_coords)
            return
        io = ZarrV2IO(None)
        assert expected_return == io.format_path(scale_key, chunk_coords)


def test_zarrio_roundtrip(tmpdir):
    accessor = get_accessor_for_url(str(tmpdir))
    accessor.store_file(".zgroup", b'{"zarr_format": 2}')
    zattrs = {"multiscales": [{"datasets": [{"path": "foo0"}]}]}
    zzarray0 = {
        "dimension_separator": "/",
        "chunks": [8, 3, 7],
        "compressor": {"id": "gzip"},
        "dtype": "uint16",
    }

    accessor.store_file(".zattrs", json.dumps(zattrs).encode("utf-8"))
    accessor.store_file("foo0/.zarray", json.dumps(zzarray0).encode("utf-8"))

    dummy_chunk = np.arange(8 * 3 * 7, dtype="uint16").reshape(1, 7, 3, 8)
    io = ZarrV2IO(accessor)
    io.write_chunk(dummy_chunk, "foo0", (0, None, 0, "foobar", 0, 15))

    retrieved_chunk = io.read_chunk("foo0", (0, 8, 0, 3, 0, 7))
    assert np.array_equal(dummy_chunk, retrieved_chunk)
