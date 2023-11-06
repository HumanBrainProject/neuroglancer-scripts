import pytest
from unittest.mock import patch, MagicMock, call, PropertyMock
from neuroglancer_scripts.sharded_file_accessor import (
    InMemByteArray,
    OnDiskByteArray,
    OnDiskBytesDict,
    ShardSpec,
    MiniShard,
    ShardedIOError,
    Shard,
    ShardedScale,
    ShardVolumeSpec,
    ShardedFileAccessor,
)
import numpy as np
import pathlib
import json


def test_ondisk_bytes_dict():
    val = OnDiskBytesDict()
    val['foo'] = b"bazz"
    assert 'foo' in val
    assert val['foo'] == b'bazz'
    assert val.get("foo") == b'bazz'
    assert list(val.keys()) == ["foo"]
    assert len(val) == 1

    val['foo'] = b"bar"
    assert val['foo'] == b'bar'
    assert val.get("foo") == b'bar'
    assert list(val.keys()) == ["foo"]
    assert len(val) == 1

    assert val.pop('foo') == b'bar'

    assert 'foo' not in val
    assert len(val) == 0
    assert list(val.keys()) == []

    with pytest.raises(Exception):
        val['foo']
    with pytest.raises(Exception):
        val.get("foo")
    with pytest.raises(Exception):
        val.pop("foo")


def test_inmem_byte_array():
    buf = InMemByteArray()
    buf += b'abc'
    assert len(buf) == 3
    assert b"".join(buf) == b'abc'


def test_on_disk_byte_array():
    buf = OnDiskByteArray()
    buf = b"abc" + buf
    buf = buf + b"def"
    assert len(buf) == 6
    assert b"".join(buf) == b"abcdef"


# MiniShard
@pytest.fixture
def shard_spec_2_2_2():
    return ShardSpec(minishard_bits=2, shard_bits=2, preshift_bits=2)


@pytest.fixture
def shard_spec_1_1_1():
    return ShardSpec(minishard_bits=1, shard_bits=1, preshift_bits=1)


store_cmc_args = [
    ([], 0b001100, True, False),
    ([], 0b001101, False, True),
    ([(bytearray(b"foo-bar"), np.uint64(0b001100))], 0b001101, True, False),
]


@pytest.mark.parametrize("pre_calls, cmc, expect_append_called, exp_apd",
                         store_cmc_args)
def test_minishard_store_cmc_chunk(shard_spec_2_2_2: ShardSpec, pre_calls, cmc,
                                   expect_append_called,
                                   exp_apd):
    shard = MiniShard(shard_spec_2_2_2, strategy="in memory")

    # Accessing next_cmc before calling store_cmc_chunk will raise
    with pytest.raises(ShardedIOError):
        print(shard.next_cmc)

    for pre_call in pre_calls:
        shard.store_cmc_chunk(*pre_call)

    len_before = len(shard._chunk_buffer)

    with patch.object(shard, "append") as append_mock:
        shard.store_cmc_chunk(b"foo bar", np.uint64(cmc))

        if expect_append_called:
            assert append_mock.called
        else:
            assert not append_mock.called

    len_after = len(shard._chunk_buffer)

    assert exp_apd == (len_before != len_after)

    # Accessing next_cmc after calling store_cmc_chunk will be ok
    print(shard.next_cmc)


append_args = [
    (
        [(b"foo-bar", np.uint64(0x0))],
        # databytearray, _last_chunk_id, header, _appended
        b"foo-bar",
        np.uint64(0x0),
        np.array([0, 0, 7], dtype=np.uint64),
        np.uint64(1)
    ),
    (
        [(b"foo", np.uint64(0x2)), (b"bar", np.uint64(0x3))],
        # databytearray, _last_chunk_id, header, _appended
        b"foobar",
        np.uint64(0x3),
        np.array([2, 0, 3, 1, 0, 3], dtype=np.uint64),
        np.uint64(2)
    ),
]


@pytest.mark.parametrize("append_calls, ex_ba, ex_last_chid, ex_hdr, ex_apd",
                         append_args)
def test_append(shard_spec_2_2_2: ShardSpec, append_calls, ex_ba, ex_last_chid,
                ex_hdr,
                ex_apd):
    shard = MiniShard(shard_spec_2_2_2)
    for c in append_calls:
        shard.append(*c)

    assert b"".join([d for d in shard.databytearray]) == ex_ba
    assert shard._last_chunk_id == ex_last_chid
    assert shard.header.tolist() == ex_hdr.tolist()
    assert int(shard._appended) == int(ex_apd)


can_be_appended_args = [
    (0b001100, 0, True, False),
    (0b001101, 0, False, False),
    (0b001100, 1, None, True),
]


@pytest.mark.parametrize("cmc, appended, can_be_appended, raise_flag",
                         can_be_appended_args)
def test_can_be_appended(shard_spec_2_2_2: ShardSpec, cmc, appended,
                         can_be_appended, raise_flag):
    shard = MiniShard(shard_spec_2_2_2, strategy="in memory")
    shard._appended = np.uint64(appended)
    spec = shard.shard_spec
    shard.masked_bits = ((spec.minishard_mask | spec.shard_mask)
                         << spec.preshift_bits) & np.uint64(cmc)
    if raise_flag:
        with pytest.raises(RuntimeError):
            shard.can_be_appended(cmc)
        return
    assert shard.can_be_appended(cmc) == can_be_appended


flush_buffer_args = [
    ([(b"foo", np.uint64(0b111100))],
     {np.uint64(0b111101): b"bar", np.uint64(0b111110): b"bar"},
     2,
     False),
    ([(b"foo", np.uint64(0b111100))],
     {np.uint64(0b111101): b"bar", np.uint64(0b111111): b"bar"},
     1,
     False),
    ([(b"foo", np.uint64(0b111100))],
     {np.uint64(0b111110): b"bar", np.uint64(0b111111): b"bar"},
     0,
     False),
    ([(b"foo", np.uint64(0b111100)), (b"foo", np.uint64(0b111101))],
     {np.uint64(0b111100): b"bar"},
     None,
     True),
]


@pytest.mark.parametrize("pre_store_cmc,chunk_buffer,exp_inserts,error_flag",
                         flush_buffer_args)
def test_flush_buffer(shard_spec_2_2_2: ShardSpec, pre_store_cmc, chunk_buffer,
                      exp_inserts,
                      error_flag):
    shard = MiniShard(shard_spec_2_2_2, strategy="in memory")
    for call_args in pre_store_cmc:
        shard.store_cmc_chunk(*call_args)
    assert len(shard._chunk_buffer) == 0

    inserted_before = shard._appended
    shard._chunk_buffer = chunk_buffer
    if error_flag:
        with pytest.raises(ShardedIOError):
            shard.flush_buffer()
        return
    shard.flush_buffer()
    inserted_after = shard._appended
    assert exp_inserts == (inserted_after - inserted_before)


close_args = [
    ([(b"foo", np.uint64(0b111100))],
     {np.uint64(0b111101): b"bar", np.uint64(0b111110): b"bar"}, 3),
    ([(b"foo", np.uint64(0b111100))],
     {np.uint64(0b111101): b"bar", np.uint64(0b111111): b"bar"}, 4),
    ([(b"foo", np.uint64(0b111100))],
     {np.uint64(0b111110): b"bar", np.uint64(0b111111): b"bar"}, 4),
]


@pytest.mark.parametrize("pre_store_cmc,chunk_buffer,exp_inserts", close_args)
def test_minishard_close(shard_spec_2_2_2: ShardSpec, pre_store_cmc,
                         chunk_buffer,
                         exp_inserts):
    shard = MiniShard(shard_spec_2_2_2, strategy="in memory")
    for call_args in pre_store_cmc:
        shard.store_cmc_chunk(*call_args)
    assert len(shard._chunk_buffer) == 0
    shard._chunk_buffer = chunk_buffer
    shard.close()
    assert shard._appended == exp_inserts


# Shard
@pytest.mark.parametrize("write_files, readable, legacy", [
    ((), False, False),
    (("{shard_key_str}.shard",), True, False),
    (("{shard_key_str}.index", "{shard_key_str}.data"), True, True)
])
def test_init_shard(tmpdir, shard_spec_2_2_2, write_files, readable, legacy):
    Shard.get_minishards_offsets = MagicMock(return_value=[])
    shard_key_str = "1"
    shard_key = np.uint64(1)
    for f in write_files:
        path_to_write = tmpdir / f.format(shard_key_str=shard_key_str)
        with open(path_to_write, "wb") as fp:
            fp.write(b"foo")
    shard = Shard(tmpdir, shard_key, shard_spec_2_2_2)
    assert shard.can_read_cmc == readable
    assert shard.is_legacy == legacy
    if readable:
        shard.get_minishards_offsets.assert_called_once()
    else:
        shard.get_minishards_offsets.assert_not_called()


@patch.object(Shard, "get_minishards_offsets", return_value=[0, 5, 5, 15])
@patch.object(Shard, "read_bytes", return_value=(
    np.array([0, 1, 2, 3, 4, 5], dtype=np.uint64).tobytes()
))
def test_init_shard_minishards(read_bytes_mock, get_minishards_offsets_mock,
                               tmpdir,
                               shard_spec_2_2_2):
    Shard.get_minishard_key = MagicMock()
    Shard.get_minishard_key.side_effect = [
        "foo",
        "bar",
    ]

    shard_key_str = "1"
    shard_key = np.uint64(1)

    with open(pathlib.Path(tmpdir) / f"{shard_key_str}.shard", "wb") as fp:
        fp.write(b"foo bar")

    shard = Shard(tmpdir, shard_key, shard_spec_2_2_2)

    read_bytes_mock.assert_has_calls([
        call(0 + shard.header_byte_length, 5),
        call(5 + shard.header_byte_length, 10),
    ])
    assert "foo" in shard.ro_minishard_dict
    assert "bar" in shard.ro_minishard_dict


header_bytes_length = int(2 ** 2 * 16)


@pytest.fixture
def legacy_shard(tmpdir, shard_spec_2_2_2: ShardSpec):
    shard_key_str = "1"
    shard_key = np.uint64(1)
    header_bytes_length = int(2 ** shard_spec_2_2_2.minishard_bits * 16)
    body_length = 5000

    with open(pathlib.Path(tmpdir) / f"{shard_key_str}.index", "wb") as fp:
        fp.write(b"h" * header_bytes_length)

    with open(pathlib.Path(tmpdir) / f"{shard_key_str}.data", "wb") as fp:
        fp.write(b"b" * body_length)

    with patch.object(Shard, "get_minishards_offsets", return_values=[]):
        shard = Shard(tmpdir, shard_key, shard_spec_2_2_2)
        yield shard, b"h", b"b"


@pytest.fixture
def modern_shard(tmpdir, shard_spec_2_2_2: ShardSpec):
    shard_key_str = "1"
    shard_key = np.uint64(1)
    header_bytes_length = int(2 ** shard_spec_2_2_2.minishard_bits * 16)
    body_length = 5000

    with open(pathlib.Path(tmpdir) / f"{shard_key_str}.shard", "wb") as fp:
        fp.write(b"i" * header_bytes_length)
        fp.write(b"j" * body_length)

    with patch.object(Shard, "get_minishards_offsets", return_values=[]):
        shard = Shard(tmpdir, shard_key, shard_spec_2_2_2)
        yield shard, b"i", b"j"


@pytest.mark.parametrize("use_shard_str", ["legacy_shard", "modern_shard"])
@pytest.mark.parametrize("byte_start, byte_length, expected_result", [
    (0, 5, "header"),
    (header_bytes_length - 5, 5, "header"),
    (header_bytes_length, 5, "body"),
])
def test_read_bytes(use_shard_str, byte_start, byte_length, expected_result,
                    request):
    shard, hb, bb = request.getfixturevalue(use_shard_str)
    expected_bytes = None

    shard.can_read_cmc = False
    with pytest.raises(ShardedIOError):
        shard.read_bytes(0, 1)
    shard.can_read_cmc = True

    if expected_result == "header":
        expected_bytes = hb * byte_length

    if expected_result == "body":
        expected_bytes = bb * byte_length

    if expected_bytes is None:
        raise Exception(f"expected result is ill defined: {expected_result!r}")

    result = shard.read_bytes(byte_start, byte_length)
    assert result == expected_bytes


@pytest.fixture
def writable_shard(tmpdir, shard_spec_2_2_2):
    shard_key_str = "1"
    shard_key = np.uint64(1)
    return Shard(tmpdir, shard_key, shard_spec_2_2_2), shard_key_str


dummy_value = "foo-bar"


@pytest.mark.parametrize("key_exist", [True, False])
@patch.object(Shard, "get_minishard_key", return_value=dummy_value)
@patch.object(MiniShard, "store_cmc_chunk")
def test_shard_store_cmc_chunk(ms_store_cmc_chunk_mock, get_minishard_key_mock,
                               key_exist,
                               shard_spec_2_2_2: ShardSpec,
                               writable_shard):

    # setup
    modern_shard, *_ = writable_shard
    minishard = MiniShard(shard_spec_2_2_2, strategy="in memory")
    modern_shard.minishard_dict = MagicMock()
    modern_shard.minishard_dict.__getitem__.return_value = minishard
    modern_shard.minishard_dict.__contains__.return_value = key_exist

    # act
    modern_shard.store_cmc_chunk(b"foo", np.uint64(123))

    # assertions
    modern_shard.minishard_dict.__contains__.assert_called_once_with(
        dummy_value)
    modern_shard.minishard_dict.__getitem__.assert_called_once_with(
        dummy_value)
    get_minishard_key_mock.assert_called_once()

    if key_exist:
        modern_shard.minishard_dict.__setitem__.assert_not_called()
    else:
        modern_shard.minishard_dict.__setitem__.assert_called_once()

    ms_store_cmc_chunk_mock.assert_called_once_with(b"foo", np.uint64(123))

    # test flag
    modern_shard.can_write_cmc = False
    with pytest.raises(ShardedIOError):
        modern_shard.store_cmc_chunk(b"foo", np.uint64(123))
    modern_shard.can_write_cmc = True


@pytest.mark.parametrize("key_exist", [True, False])
@patch.object(Shard, "get_minishard_key", return_value=dummy_value)
@patch.object(MiniShard, "fetch_cmc_chunk")
def test_fetch_cmc_chunk(ms_fetch_cmc_chunk_mock, get_minishard_key_mock,
                         key_exist,
                         shard_spec_2_2_2: ShardSpec,
                         writable_shard):

    modern_shard, *_ = writable_shard
    minishard = MiniShard(shard_spec_2_2_2, strategy="in memory")
    modern_shard.ro_minishard_dict = MagicMock()
    modern_shard.ro_minishard_dict.__getitem__.return_value = minishard
    modern_shard.ro_minishard_dict.__contains__.return_value = key_exist
    ms_fetch_cmc_chunk_mock.return_value = b"foo-bar"

    if not key_exist:
        with pytest.raises(AssertionError):
            modern_shard.fetch_cmc_chunk(np.uint64(123))
        modern_shard.ro_minishard_dict.__contains__.assert_called_once_with(
            dummy_value)
        get_minishard_key_mock.assert_called_once_with(np.uint64(123))
        ms_fetch_cmc_chunk_mock.assert_not_called()
        return

    assert modern_shard.fetch_cmc_chunk(np.uint64(123)) == b"foo-bar"
    ms_fetch_cmc_chunk_mock.assert_called_once_with(np.uint64(123))
    modern_shard.ro_minishard_dict.__getitem__.assert_called_once_with(
        dummy_value)


@patch.object(MiniShard, 'offset', new_callable=PropertyMock, return_value=0)
@patch.object(MiniShard, 'close', return_value=1)
def test_shard_close(minishard_close_mock, offset_mock, tmpdir,
                     shard_spec_1_1_1):
    shard_key = np.uint64(1)
    shard = Shard(tmpdir, shard_key, shard_spec_1_1_1)
    shard.dirty = True

    minishard0 = MiniShard(shard_spec_1_1_1, strategy="in memory")
    minishard1 = MiniShard(shard_spec_1_1_1, strategy="in memory")
    shard.minishard_dict = {
        "b": minishard1,
        "a": minishard0,
    }
    assert list(shard.minishard_dict.keys()) == ["b", "a"]

    minishard0.databytearray = [b"0", b"0"]
    minishard0.header = np.array([0, 1, 2, 3, 4, 5], dtype=np.uint64)

    minishard1.databytearray = [b"1", b"1"]
    minishard1.header = np.array([6, 7, 8, 9, 10, 11], dtype=np.uint64)

    shard.close()

    assert minishard_close_mock.call_count == 2

    with open(pathlib.Path(tmpdir) / "1.shard", "rb") as fp:
        b = fp.read()

    hdr_size = int((2**shard.shard_spec.minishard_bits) * 16)
    msh_hdr_size = 6 * 8  # only true for raw encoders

    # check file length
    assert len(b) == (hdr_size + 4 + (12 * 8))

    # check header
    sh_hdr = np.array([
        4, 6 * 8 + 4,  # minishard1 hdr offset_start, offset_end
        6 * 8 + 4, 6 * 8 * 2 + 4 ,  # minishard2 hdr offset_start, offset_end
    ], dtype=np.uint64).tobytes()
    assert b[:len(sh_hdr)] == sh_hdr

    # check volume, esp that dict is sorted
    assert b[len(sh_hdr): len(sh_hdr)+4] == b"0011"

    # check minishard hdrs
    offset0 = len(sh_hdr) + 4
    assert (b[offset0 : offset0 + msh_hdr_size]
            == np.array([0, 3, 1, 4, 2, 5], dtype=np.uint64).tobytes())

    offset1 = offset0 + msh_hdr_size
    assert (b[offset1 : offset1 + msh_hdr_size]
            == np.array([6, 9, 7, 10, 8, 11], dtype=np.uint64).tobytes())


def test_shard_close_toomany_minishards(tmpdir, shard_spec_1_1_1):

    shard_key = np.uint64(1)
    shard = Shard(tmpdir, shard_key, shard_spec_1_1_1)
    shard.dirty = True

    minishard0 = MiniShard(shard_spec_1_1_1, strategy="in memory")
    minishard1 = MiniShard(shard_spec_1_1_1, strategy="in memory")
    minishard2 = MiniShard(shard_spec_1_1_1, strategy="in memory")

    minishard0.databytearray = [b"0", b"0"]
    minishard0.header = np.array([0, 1, 2, 3, 4, 5], dtype=np.uint64)

    minishard1.databytearray = [b"0", b"0"]
    minishard1.header = np.array([0, 1, 2, 3, 4, 5], dtype=np.uint64)

    minishard2.databytearray = [b"0", b"0"]
    minishard2.header = np.array([0, 1, 2, 3, 4, 5], dtype=np.uint64)

    shard.minishard_dict = {
        "c": minishard2,
        "b": minishard1,
        "a": minishard0,
    }
    with pytest.raises(ShardedIOError):
        shard.close()


@patch.object(MiniShard, 'offset', new_callable=PropertyMock, return_value=0)
@patch.object(MiniShard, 'close', return_value=1)
def test_shard_close_toofew_minishards(minishard_close_mock, offset_mock,
                                       tmpdir, shard_spec_1_1_1):

    shard_key = np.uint64(1)
    shard = Shard(tmpdir, shard_key, shard_spec_1_1_1)
    shard.dirty = True

    minishard0 = MiniShard(shard_spec_1_1_1, strategy="in memory")
    shard.minishard_dict = {
        "a": minishard0,
    }

    minishard0.databytearray = [b"0", b"0"]
    minishard0.header = np.array([0, 1, 2, 3, 4, 5], dtype=np.uint64)

    shard.close()
    assert minishard_close_mock.call_count == 1

    with open(pathlib.Path(tmpdir) / "1.shard", "rb") as fp:
        b = fp.read()

    hdr_size = int((2**shard.shard_spec.minishard_bits) * 16)
    msh_hdr_size = 6 * 8  # only true for raw encoders

    # check file length
    assert len(b) == (hdr_size + 2 + (12 * 4))

    # check header
    sh_hdr = np.array([
        2, 6 * 8 + 2,  # minishard1 hdr offset_start, offset_end
        6 * 8 + 2, 6 * 8 + 2 ,  # empty shard, same start & end
    ], dtype=np.uint64).tobytes()
    assert b[:len(sh_hdr)] == sh_hdr

    # check volume, esp that dict is sorted
    assert b[len(sh_hdr): len(sh_hdr)+2] == b"00"

    # check minishard hdrs
    offset0 = len(sh_hdr) + 2
    assert (b[offset0 : offset0 + msh_hdr_size]
            == np.array([0, 3, 1, 4, 2, 5], dtype=np.uint64).tobytes())


def test_shard_close_not_dirty(tmpdir, shard_spec_1_1_1):
    shard_key = np.uint64(1)
    shard = Shard(tmpdir, shard_key, shard_spec_1_1_1)
    shard.file_path = MagicMock()
    shard.file_path.parent.mkdir.side_effect = Exception("foobar")
    with pytest.raises(Exception):
        shard.dirty = True
        shard.close()
    shard.dirty = False
    shard.close()


# ShardedScale
@pytest.fixture
def sharded_scale(tmpdir, shard_spec_2_2_2):
    vol_spec = ShardVolumeSpec([64, 64, 64], [128, 128, 128])
    return ShardedScale(tmpdir, "5mm", shard_spec_2_2_2, vol_spec)


@pytest.mark.parametrize("key_exists", [True, False])
def test_get_shard(sharded_scale: ShardedScale, key_exists: bool):

    # setup
    sharded_scale.shard_dict = MagicMock()
    sharded_scale.shard_dict.__contains__.return_value = key_exists
    sharded_scale.shard_dict.__getitem__.return_value = "foo"

    # act
    return_val = sharded_scale.get_shard(np.uint64(123))

    # assertions
    assert return_val == "foo"
    sharded_scale.shard_dict.__contains__.assert_called_once_with(
        np.uint64(123))
    if key_exists:
        sharded_scale.shard_dict.__setitem__.assert_not_called()
    else:
        sharded_scale.shard_dict.__setitem__.assert_called_once()
    sharded_scale.shard_dict.__getitem__.assert_called_once_with(
        np.uint64(123))


def test_sharded_scale_close(sharded_scale: ShardedScale):

    mock1 = MagicMock()
    mock2 = MagicMock()
    sharded_scale.shard_dict = {
        "foo": mock1,
        "bar": mock2,
    }

    sharded_scale.close()

    mock1.close.__assert_called_once()
    mock2.close.__assert_called_once()


# ShardedFileAccessor
shard_scale_info = {
    "scales": [
        {
            "key": "20mm",
            "size": [256, 256, 256],
            "chunk_sizes": [
                [64, 64, 64]
            ],
            "sharding": {
                "@type": "neuroglancer_uint64_sharded_v1",
                "shard_bits": 2,
                "minishard_bits": 2
            }
        },
        {
            "key": "40mm",
            "size": [128, 128, 128],
            "chunk_sizes": [
                [64, 64, 64]
            ],
            "sharding": {
                "@type": "neuroglancer_uint64_sharded_v1",
                "shard_bits": 2,
                "minishard_bits": 2
            }
        }
    ]
}


@pytest.fixture
def faccessor_r(tmpdir):
    with open(pathlib.Path(tmpdir) / "info", "w") as fp:
        json.dump(shard_scale_info, fp=fp)
    return ShardedFileAccessor(tmpdir)


@pytest.fixture
def faccessor_w(tmpdir):
    with open(tmpdir / "info", "w") as fp:
        json.dump({
            "scales": [
                {
                    "size": [256, 256, 256],
                    "chunk_sizes": [[64, 64, 64]],
                    "key": "20mm",
                    "sharding": {
                        "@type": "neuroglancer_uint64_sharded_v1",
                        "minishard_bits": 2,
                        "shard_bits": 2,
                        "hash": "identity",
                        "minishard_index_encoding": "raw",
                        "data_encoding": "raw",
                        "preshift_bits": 0
                    }
                },
                {
                    "size": [128, 128, 128],
                    "chunk_sizes": [[64, 64, 64]],
                    "key": "40mm",
                    "sharding": {
                        "@type": "neuroglancer_uint64_sharded_v1",
                        "minishard_bits": 2,
                        "shard_bits": 2,
                        "hash": "identity",
                        "minishard_index_encoding": "raw",
                        "data_encoding": "raw",
                        "preshift_bits": 0
                    }
                }
            ]
        }, fp=fp)
    yield ShardedFileAccessor(tmpdir)


def test_readable(faccessor_r: ShardedFileAccessor):
    assert faccessor_r.can_read


def test_writable(faccessor_w: ShardedFileAccessor):
    assert faccessor_w.can_write


def test_file_exists(faccessor_w: ShardedFileAccessor, tmpdir):
    with open(pathlib.Path(tmpdir) / "exists", "w"):
        ...
    assert faccessor_w.file_exists("exists")
    assert not faccessor_w.file_exists("notexists")
    assert not faccessor_w.file_exists("dir/notexists")


def test_fetch_file(faccessor_w: ShardedFileAccessor, tmpdir):
    with open(pathlib.Path(tmpdir) / "file", "wb") as fp:
        fp.write(b"foo-bar")
    assert faccessor_w.fetch_file("file") == b"foo-bar"


def test_store_file(faccessor_w: ShardedFileAccessor, tmpdir):
    with open(pathlib.Path(tmpdir) / "exists", "wb") as fp:
        fp.write(b"abc")

    faccessor_w.store_file("non_exist", b"bazzbizz", False)
    with open(pathlib.Path(tmpdir) / "non_exist", "rb") as fp:
        assert fp.read() == b"bazzbizz"

    with pytest.raises(IOError):
        faccessor_w.store_file("exists", b"bazzbizz", False)

    with open(pathlib.Path(tmpdir) / "exists", "rb") as fp:
        assert fp.read() == b"abc"

    faccessor_w.store_file("exists", b"bazzbizz", True)

    with open(pathlib.Path(tmpdir) / "exists", "rb") as fp:
        assert fp.read() == b"bazzbizz"


def test_fetch_chunk_error(faccessor_r: ShardedFileAccessor):
    with pytest.raises(Exception):
        faccessor_r.fetch_chunk("10mm", (0, 0, 0, 0, 0, 0))


@pytest.mark.parametrize("chunk_coords", [
    (0, 64, 0, 64, 0, 64),
    (0, None, 128, None, 0, None),
])
@pytest.mark.parametrize("key", ["20mm", "40mm"])
@patch.object(ShardedScale, "fetch_chunk")
def test_fetch_chunk_success(fetch_chunk_m, faccessor_r: ShardedFileAccessor,
                             key,
                             chunk_coords):
    fetch_chunk_m.return_value = b"foo-bar"

    return_value = faccessor_r.fetch_chunk(key, chunk_coords)

    fetch_chunk_m.assert_called_once_with(chunk_coords)
    assert return_value == b"foo-bar"


@pytest.mark.parametrize("key", ["10mm", None])
@pytest.mark.parametrize("chunk_coords", [(0, 64, 63, 128, 64, 128), None])
def test_store_chunk_error(faccessor_w: ShardedFileAccessor,
                           key,
                           chunk_coords):
    if key is None and chunk_coords is None:
        return
    key = key or "20mm"
    chunk_coords = chunk_coords or (0, 0, 0)
    with pytest.raises(Exception):
        faccessor_w.store_chunk(b"123", key, chunk_coords)


@pytest.mark.parametrize("key", ["20mm", "40mm"])
@pytest.mark.parametrize("chunk_coords", [
    (0, 64, 0, 64, 0, 64),
    # x/y/z_end is no longer relevant in determining grid coordinate
    (0, "bar", 0, 17.4, 0, []),
    (64, 128, 64, 128, 0, 64),
])
@patch.object(ShardedScale, "store_chunk", return_value=None)
def test_store_chunk_success(scale_store_chunk_mock,
                             faccessor_w: ShardedFileAccessor,
                             key,
                             chunk_coords):
    faccessor_w.store_chunk(b"123", key, chunk_coords)
    scale_store_chunk_mock.assert_called_once_with(b"123", chunk_coords)


def test_accessor_close(faccessor_w: ShardedFileAccessor, tmpdir):
    mock1 = MagicMock()
    mock2 = MagicMock()

    mock1.to_json.return_value = "tojson_1"
    mock2.to_json.return_value = "tojson_2"
    faccessor_w.shard_dict = {
        "1": mock1,
        "2": mock2,
    }

    faccessor_w.close()

    mock1.close.assert_called_once()
    mock2.close.assert_called_once()

    mock1.to_json.assert_called_once()
    mock2.to_json.assert_called_once()
