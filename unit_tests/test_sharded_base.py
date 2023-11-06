import pytest
from collections import namedtuple
import numpy as np
import zlib
from unittest.mock import patch, MagicMock

from neuroglancer_scripts.sharded_base import (
    ShardVolumeSpec, ShardSpec, CMCReadWrite, ShardedIOError,
    ReadableMiniShardCMC, ShardedScaleBase, ShardedAccessorBase, ShardCMC
)


ShardSpecArg = namedtuple("ShardSpecArg", ["minishard_bits", "shard_bits",
                                           "hash",
                                           "minishard_index_encoding",
                                           "data_encoding",
                                           "preshift_bits"])


ExpRes = namedtuple("ExpRes", "shard_mask minishard_mask preshift_mask error")
ExpRes.__new__.__defaults__ = (None,) * len(ExpRes._fields)

shard_spec_args = [
    (ShardSpecArg(1, 1, "identity", "raw", "raw", 0), ExpRes(0b10, 0b1)),

    (ShardSpecArg(2, 2, "identity", "gzip", "raw", 0), ExpRes(0b1100, 0b11)),
    (ShardSpecArg(2, 2, "identity", "raw", "gzip", 0), ExpRes(0b1100, 0b11)),
    (ShardSpecArg(2, 2, "identity", "gzip", "gzip", 0), ExpRes(0b1100, 0b11)),

    (ShardSpecArg(2, 2, "identity", "raw", "raw", -1), ExpRes(error=True)),
    (ShardSpecArg(-1, 2, "identity", "raw", "raw", 0), ExpRes(error=True)),
    (ShardSpecArg(2, -1, "identity", "raw", "raw", 0), ExpRes(error=True)),
    (
        ShardSpecArg(2, 2, "murmurhash3_x86_128", "raw", "raw", 0),
        ExpRes(error=True)
    ),
    (ShardSpecArg(2, 2, "identity", "zlib", "raw", 0), ExpRes(error=True)),
    (ShardSpecArg(2, 2, "identity", "raw", "foobar", 0), ExpRes(error=True)),
]


@pytest.mark.parametrize("shard_spec_arg, expected_result", shard_spec_args)
def test_shard_spec(shard_spec_arg: ShardSpecArg, expected_result: ExpRes):
    if expected_result.error:

        with pytest.raises(Exception):
            ShardSpec(*shard_spec_arg)
        return

    shard_spec = ShardSpec(*shard_spec_arg)
    print("shard_spec.minishard_mask", shard_spec.minishard_mask)
    assert shard_spec.minishard_mask == expected_result.minishard_mask
    assert shard_spec.shard_mask == expected_result.shard_mask


class Foo(CMCReadWrite):
    pass


shard_key_args = range(1, 8)


@pytest.mark.parametrize("shard_bits", shard_key_args)
def test_get_shard_key(shard_bits):
    shard_spec = ShardSpec(minishard_bits=0, shard_bits=shard_bits)
    expected_keys = {
        np.uint64(i)
        for i in range(2 ** int(shard_bits))
    }
    cmc_rw = Foo(shard_spec)
    assert {
        cmc_rw.get_shard_key(np.uint64(num))
        for num in range(1024)
    } == expected_keys


text = b'foo bar'
compressed_text = zlib.compress(text)

encode_decode_args = [
    ("raw", (text, text), (text, text)),
    ("gzip", (text, compressed_text), (compressed_text, text)),
]


@pytest.mark.parametrize("enc_type, encoding, decoding", encode_decode_args)
def test_data_encode_decode(enc_type, encoding, decoding):
    enc_from, enc_to = encoding
    dec_from, dec_to = decoding
    spec = ShardSpec(0, 0, "identity", enc_type, enc_type, 0)

    encoded = spec.data_encoder(enc_from)
    assert encoded == enc_to
    assert spec.data_decoder(encoded) == enc_from

    decoded = spec.data_decoder(dec_from)
    assert decoded == dec_to
    assert spec.data_encoder(decoded) == dec_from

    encoded = spec.index_encoder(enc_from)
    assert encoded == enc_to
    assert spec.index_decoder(encoded) == enc_from

    decoded = spec.index_decoder(dec_from)
    assert decoded == dec_to
    assert spec.index_encoder(decoded) == dec_from


v0 = (
    (0, 0),
    (1, 1),
    (4, 4),
)


to_dict_args = [
    (0, 0, "identity", "raw", "raw", 0),
    (0, 0, "identity", "gzip", "gzip", 0),
    (1, 1, "identity", "raw", "raw", 1),
    (4, 4, "identity", "raw", "raw", 4),
]


@pytest.mark.parametrize("minishard_bits, shard_bits, hash, "
                         "minishard_index_encoding, data_encoding,"
                         "preshift_bits", to_dict_args)
def test_to_dict(minishard_bits, shard_bits, hash, minishard_index_encoding,
                 data_encoding, preshift_bits):
    spec = ShardSpec(minishard_bits, shard_bits, hash,
                     minishard_index_encoding, data_encoding, preshift_bits)
    spec_dict = spec.to_dict()
    for key in [
        "minishard_bits", "shard_bits", "hash", "minishard_index_encoding",
        "data_encoding", "preshift_bits"
    ]:
        assert spec_dict.get(key) == eval(key)


shard_mask_args = [
    (0, 0, 0, 0b0),
    (0, 1, 0, 0b1),
    (1, 1, 0, 0b10),
    (1, 2, 0, 0b110),

    # preshift bits do not contribute to shard_mask
    (0, 1, 1, 0b1),
    (1, 1, 1, 0b10),
]


@pytest.mark.parametrize("minishard_bits, shard_bits, preshift_bits,"
                         "expected_shard_mask", shard_mask_args)
def test_shard_mask(minishard_bits, shard_bits, preshift_bits,
                    expected_shard_mask):
    spec = ShardSpec(minishard_bits=minishard_bits, shard_bits=shard_bits,
                     preshift_bits=preshift_bits)
    assert spec.shard_mask == np.uint64(expected_shard_mask)


mini_shard_mask_args = [
    (0, 0, 0, 0b0),
    (1, 0, 0, 0b1),
    (2, 0, 0, 0b11),

    # preshift bits/shard bitz do not contribute to minishard_mask
    (1, 1, 0, 0b1),
    (1, 0, 1, 0b1),
    (1, 1, 1, 0b1),
]


@pytest.mark.parametrize("minishard_bits, shard_bits, preshift_bits, "
                         "expected_shard_mask", mini_shard_mask_args)
def test_minishard_mask(minishard_bits, shard_bits, preshift_bits,
                        expected_shard_mask):
    spec = ShardSpec(minishard_bits=minishard_bits, shard_bits=shard_bits,
                     preshift_bits=preshift_bits)
    assert spec.minishard_mask == np.uint64(expected_shard_mask)


preshift_mask_args = [
    (0, 0, 0, 0b0),
    (0, 0, 1, 0b1),
    (0, 0, 2, 0b11),

    # (mini)shard_bits do not contribute to preshift mask
    (0, 1, 1, 0b1),
    (1, 0, 1, 0b1),
    (1, 1, 1, 0b1),
]


@pytest.mark.parametrize("minishard_bits, shard_bits, preshift_bits, "
                         "exp_shard_mask", preshift_mask_args)
def test_preshift_mask(minishard_bits, shard_bits, preshift_bits,
                       exp_shard_mask):
    spec = ShardSpec(minishard_bits=minishard_bits, shard_bits=shard_bits,
                     preshift_bits=preshift_bits)
    assert spec.preshift_mask == np.uint64(exp_shard_mask)


# shard volume tests
shard_volume_spec_init = [
    ((64, 64, 64), (6400, 6400, 6400), (100, 100, 100), False),
    ((64, 64, 64), (1, 6400, 6400), (1, 100, 100), False),

    ((64, 64, 64), (6400, 6400), None, True),
    ((64, 64), (6400, 6400, 6400), None, True),
    ((64, 64, 64), (6400, 6400, -6400), None, True),
    ((64, 64, 64), (6400, 6400, 6.4e3), None, True),
    ((128, 64, 64), (6400, 6400, 6400), None, True),
    ((-64, -64, -64), (6400, 6400, 6400), None, True),
    ((64, 64, 64), (640_000_000, 640_000_000, 640_000_000), None, True),
]


@pytest.mark.parametrize("chunk_sizes, sizes, grid_sizes, raise_flag",
                         shard_volume_spec_init)
def test_shard_spec_init(chunk_sizes, sizes, grid_sizes, raise_flag):
    if raise_flag:
        with pytest.raises(ShardedIOError):
            ShardVolumeSpec(chunk_sizes, sizes)
        return
    spec = ShardVolumeSpec(chunk_sizes, sizes)
    assert spec.grid_sizes == list(grid_sizes)


@pytest.fixture
def shard_volume_spec():
    return ShardVolumeSpec(
        [64, 64, 64],
        [6400, 6400, 6400]
    )


@pytest.mark.parametrize("grid_coords,expected,err_flag", [
    ((0, 0, 0), 0b0, False),
    ((1, 0, 0), 0b1, False),
    ((0, 1, 0), 0b10, False),
    ((1, 1, 0), 0b11, False),
    ((1, 1, 1), 0b111, False),
    ((2, 0, 0), 0b1000, False),
    ((2, 0, 0), 0b1000, False),
    ((-1, 0, 0), None, True),
    ((0, -1, 0), None, True),
    ((0, 0, -1), None, True),
    ((0.5, 0, 0), None, True),
    ((101, 0, 0), None, True),
])
def test_compressed_morton_code(shard_volume_spec: ShardVolumeSpec,
                                grid_coords,
                                expected,
                                err_flag):
    if err_flag:
        with pytest.raises(ShardedIOError):
            shard_volume_spec.compressed_morton_code(grid_coords)
        return
    cmc = shard_volume_spec.compressed_morton_code(grid_coords)
    assert cmc == expected


@pytest.mark.parametrize("chunk_coords,expected,err_flag", [
    ((0, 64, 0, 64, 0, 64), 0b0, False),
    ((64, 128, 0, 64, 0, 64), 0b1, False),
    ((0, 64, 64, 128, 0, 64), 0b10, False),
    ((64, 128, 64, 128, 0, 64), 0b11, False),
    ((64, 128, 64, 128, 64, 128), 0b111, False),
    ((128, 196, 0, 64, 0, 64), 0b1000, False),
    ((0, 64, 32, 64, 0, 64), None, True),
    ((32, 64, 0, 64, 0, 64), None, True),
    ((0, 64, 0, 64, 32, 64), None, True),
])
def test_get_cmc(shard_volume_spec: ShardVolumeSpec,
                 chunk_coords,
                 expected,
                 err_flag):
    if err_flag:
        with pytest.raises(ShardedIOError):
            shard_volume_spec.get_cmc(chunk_coords)
        return
    cmc = shard_volume_spec.get_cmc(chunk_coords)
    assert cmc == expected


# CMCReadWrite
@pytest.fixture
def shard_spec_1_1_0():
    return ShardSpec(shard_bits=1, minishard_bits=1)


@pytest.fixture
def shard_spec_1_1_1():
    return ShardSpec(shard_bits=1, minishard_bits=1, preshift_bits=1)


@pytest.fixture
def shard_spec_1_2_0():
    return ShardSpec(shard_bits=1, minishard_bits=2)


@pytest.fixture
def shard_spec_1_2_1():
    return ShardSpec(shard_bits=1, minishard_bits=2, preshift_bits=1)


@pytest.fixture
def shard_spec_2_1_0():
    return ShardSpec(shard_bits=2, minishard_bits=1)


@pytest.fixture
def shard_spec_2_1_1():
    return ShardSpec(shard_bits=2, minishard_bits=1, preshift_bits=1)


header_byte_length_args = [
    ("shard_spec_1_1_0", 32),
    ("shard_spec_1_1_1", 32),
    ("shard_spec_1_2_0", 64),
    ("shard_spec_1_2_1", 64),
    ("shard_spec_2_1_0", 32),
    ("shard_spec_2_1_1", 32),
]


@pytest.mark.parametrize("sh_spec_name, exp_hdr_bytelen",
                         header_byte_length_args)
def test_get_minishard_key(sh_spec_name: str, exp_hdr_bytelen, request):
    shard_spec: ShardSpec = request.getfixturevalue(sh_spec_name)
    cmc_rw = CMCReadWrite(shard_spec)
    assert cmc_rw.header_byte_length == exp_hdr_bytelen


get_key_args = [
    ("shard_spec_1_1_0", 0b1001101, 0b1, 0b0),
    ("shard_spec_1_1_1", 0b1001101, 0b0, 0b1),
    ("shard_spec_1_2_0", 0b1001101, 0b01, 0b1),
    ("shard_spec_1_2_1", 0b1001101, 0b10, 0b1),
    ("shard_spec_2_1_0", 0b1001101, 0b1, 0b10),
    ("shard_spec_2_1_1", 0b1001101, 0b0, 0b11),
]


@pytest.mark.parametrize("sh_spec_name, cmc, exp_minsh_key, exp_sh_key",
                         get_key_args)
def test_minishard_key(sh_spec_name, cmc, exp_minsh_key,
                       exp_sh_key,
                       request):
    shard_spec: ShardSpec = request.getfixturevalue(sh_spec_name)
    cmc_rw = CMCReadWrite(shard_spec)
    assert cmc_rw.get_minishard_key(np.uint64(cmc)) == np.uint64(exp_minsh_key)
    assert cmc_rw.get_shard_key(np.uint64(cmc)) == np.uint64(exp_sh_key)


header_content_args = [
    (*existing, *(np.array([1, 2], dtype=np.uint64).tobytes(), [1, 2]))
    for existing in header_byte_length_args
]


# ShardCMC
@pytest.mark.parametrize("shard_spec_fixture_name, expected_header_byte, "
                         "read_bytes_return, expected_fn_return",
                         header_content_args)
def test_get_minishard_offset(shard_spec_fixture_name, expected_header_byte,
                              read_bytes_return,
                              expected_fn_return,
                              request):
    class DummyShardCMC(ShardCMC):
        def file_exists(self, filepath) -> bool:
            return False
    shard_spec: ShardSpec = request.getfixturevalue(shard_spec_fixture_name)
    cmc_rw = DummyShardCMC(np.uint64(1), shard_spec)
    cmc_rw.can_read_cmc = True
    with patch.object(cmc_rw, "read_bytes",
                      return_value=read_bytes_return) as read_bytes_mock:
        assert cmc_rw.get_minishards_offsets().tolist() == expected_fn_return
        read_bytes_mock.assert_called_once_with(0, expected_header_byte)


# ReadableMiniShardCMC

def get_parent_shard(minishard_bits=1, shard_bits=1, can_read=True):
    spec = ShardSpec(minishard_bits=minishard_bits, shard_bits=shard_bits)
    inst = CMCReadWrite(shard_spec=spec)
    inst.can_read_cmc = can_read
    return inst


def get_header_buffer(header_content):
    assert (
        isinstance(header_content, list)
        and all(isinstance(v, int) for v in header_content)
    )
    return np.array(header_content, dtype=np.uint64).tobytes()


readableminishard_init_args = [
    (get_parent_shard(can_read=True), get_header_buffer([1, 2, 3, 4, 5, 6]),
     2,
     False),
    (get_parent_shard(can_read=True),
     get_header_buffer([1, 2, 3, 4, 5, 6, 7, 8, 9]),
     3,
     False),
    (get_parent_shard(can_read=False), get_header_buffer([1, 2, 3, 4, 5, 6]),
     None,
     True),
    (get_parent_shard(can_read=True),
     get_header_buffer([1, 2, 3, 4, 5, 6, 7]),
     None,
     True),
]


@pytest.mark.parametrize("p_shard, hdr_buf, ex_num_chunks, raise_flag",
                         readableminishard_init_args)
def test_readableminishard_init(p_shard: CMCReadWrite,
                                hdr_buf: bytes,
                                ex_num_chunks,
                                raise_flag):
    if raise_flag:
        with pytest.raises(Exception):
            ReadableMiniShardCMC(p_shard, hdr_buf)
        return
    minishard_cmc = ReadableMiniShardCMC(p_shard, hdr_buf)
    assert minishard_cmc.num_chunks == ex_num_chunks


parent_shard = get_parent_shard(shard_bits=2, minishard_bits=2)
hdr_buf = get_header_buffer([
    0, 1, 4,
    1024, 0, 128,
    64, 64, 128,
])

fetch_cmc_chunk_args = [
    (0, (1024, 64), False),
    (1, (1024+64+0, 64), False),
    (5, (1024+64+0+64+128, 128), False),

    (2, None, True),
    (3, None, True),
    (4, None, True),
]


data_decoder_returnval = b"foo-bar"
read_bytes_returnval = b"read_bytes"


@pytest.mark.parametrize("cmc, read_bytes_args, raise_flag",
                         fetch_cmc_chunk_args)
@patch.object(ShardSpec, "data_decoder", return_value=data_decoder_returnval)
@patch.object(parent_shard, "read_bytes", return_value=read_bytes_returnval)
def test_fetch_cmc_chunk(read_bytes_mock, data_decoder_mock, cmc,
                         read_bytes_args,
                         raise_flag):
    minishard_cmc = ReadableMiniShardCMC(parent_shard, hdr_buf)
    if raise_flag:
        with pytest.raises(ShardedIOError):
            minishard_cmc.fetch_cmc_chunk(np.uint64(cmc))
        return

    from_b, b_length = read_bytes_args

    assert (minishard_cmc.fetch_cmc_chunk(np.uint64(cmc))
            == data_decoder_returnval)
    read_bytes_mock.assert_called_once_with(
        from_b + parent_shard.header_byte_length, b_length)
    data_decoder_mock.assert_called_once_with(read_bytes_returnval)


# ShardedScaleBase
class ShardedScale(ShardedScaleBase):
    def get_shard(self, *args, **kwargs):
        ...


@pytest.fixture
def mocked_get_shard_base():
    shard_spec_magic_mock = MagicMock()
    shard_volume_spec_magic_mock = MagicMock()
    return (
        ShardedScale("foo", shard_spec_magic_mock,
                     shard_volume_spec_magic_mock),
        shard_spec_magic_mock,
        shard_volume_spec_magic_mock,
    )


@pytest.mark.parametrize("test_arg, exp_flag", [
    ({"sharding": {"@type": "neuroglancer_uint64_sharded_v1"}}, True),
    ({"sharding": {"@type": "foo-bar"}}, False),
    ("foo", False),
    ([], False),
    ({}, False),
])
def test_is_sharded(test_arg, exp_flag):
    assert ShardedScaleBase.is_sharded(test_arg) == exp_flag


def test_store_cmc_chunk(mocked_get_shard_base):
    b = b"foo"
    shard_key = np.uint(456)
    cmc = np.uint64(123)

    base, shard_spec, svs = mocked_get_shard_base
    shard_magic_mock = MagicMock()
    get_shard_key_mock = MagicMock()
    with patch.object(base, "get_shard_key",
                      return_value=shard_key) as get_shard_key_mock:
        with patch.object(base, "get_shard",
                          return_value=shard_magic_mock) as get_shard_mock:
            base.store_cmc_chunk(b, cmc)
            get_shard_key_mock.assert_called_once_with(cmc)
            get_shard_mock.assert_called_once_with(shard_key)
            shard_magic_mock.store_cmc_chunk.assert_called_once_with(b, cmc)


def test_store_chunk(mocked_get_shard_base):
    b = b"foo"

    base, shard_spec, svs = mocked_get_shard_base
    with patch.object(base, "store_cmc_chunk") as store_cmc_chunk_mock:
        svs.get_cmc.return_value = np.uint64(789)
        base.store_chunk(b, (0, 64, 0, 64, 0, 64))
        svs.get_cmc.assert_called_once_with((0, 64, 0, 64, 0, 64))
        store_cmc_chunk_mock.assert_called_once_with(b, np.uint64(789))


def test_shard_fetch_cmc_chunk(mocked_get_shard_base):
    shard_key = np.uint(456)
    cmc = np.uint64(123)

    base, shard_spec, svs = mocked_get_shard_base
    shard_magic_mock = MagicMock()
    get_shard_key_mock = MagicMock()
    with patch.object(base, "get_shard_key",
                      return_value=shard_key) as get_shard_key_mock:
        with patch.object(base, "get_shard",
                          return_value=shard_magic_mock) as get_shard_mock:
            base.fetch_cmc_chunk(cmc)
            get_shard_key_mock.assert_called_once_with(cmc)
            get_shard_mock.assert_called_once_with(shard_key)
            shard_magic_mock.fetch_cmc_chunk.assert_called_once_with(cmc)


def test_fetch_chunk(mocked_get_shard_base):

    base, shard_spec, svs = mocked_get_shard_base
    with patch.object(base, "fetch_cmc_chunk") as fetch_cmc_chunk_mock:
        svs.get_cmc.return_value = np.uint64(789)
        base.fetch_chunk((0, 64, 0, 64, 0, 64))
        svs.get_cmc.assert_called_once_with((0, 64, 0, 64, 0, 64))
        fetch_cmc_chunk_mock.assert_called_once_with(np.uint64(789))


def test_to_json(mocked_get_shard_base):
    base, shard_spec, svs = mocked_get_shard_base
    shard_spec.to_dict.return_value = {"foo": "bar"}
    assert base.to_json() == {
        "foo": "bar",
        "@type": "neuroglancer_uint64_sharded_v1"
    }


# ShardedAccessorBase

info_setter_args = [
    (
        {
            "scales": [
                {
                    "key": "foo",
                    "sharding": {
                        "@type": "neuroglancer_uint64_sharded_v1"
                    }
                }
            ]
        } ,
        False
    ),
    (
        {
            "scales": [
                {
                    "key": f"foo{idx}",
                    "sharding": {
                        "@type": "neuroglancer_uint64_sharded_v1"
                    }
                } for idx in range(2)
            ]
        } ,
        False
    ),

    (
        {
            "scales": [
                {
                    "key": "foo",
                }
            ]
        } ,
        True
    ),

    (
        {
            "scales": [
                {
                    "key": "foo",
                    "sharding": {}
                }
            ]
        } ,
        True
    ),
    (
        {
            "scales": [
                {
                    "key": "foo",
                    "sharding": {
                        "@type": "foo-bar"
                    }
                }
            ]
        } ,
        True
    ),

    (
        {
            "scales": [
                {
                    "key": f"foo{idx}",
                    "sharding": {
                        "@type": "neuroglancer_uint64_sharded_v1"
                    } if idx % 2 == 0 else {}
                } for idx in range(2)
            ]
        } ,
        True
    ),

    ({"scales": []} , True),
    ((1, 2, 3), True),
    ([], True),
    ("info", True),
]


@pytest.mark.parametrize("info, raise_error_flag", info_setter_args)
def test_info_setter(info, raise_error_flag):

    acc = ShardedAccessorBase()

    with pytest.raises(AttributeError):
        len(acc.info)

    if raise_error_flag:
        with pytest.raises(Exception):
            acc.info = info
        return

    acc.info = info
    len(acc.info)
