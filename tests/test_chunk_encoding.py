# Copyright (c) 2018 Forschungszentrum Juelich GmbH
# Author: Yann Leprince <y.leprince@fz-juelich.de>
#
# This software is made available under the MIT licence, see LICENCE.txt.

import numpy as np
import pytest

from neuroglancer_scripts.chunk_encoding import *


@pytest.mark.parametrize("encoder_options", [
    {},
    {"jpeg_quality": 1, "jpeg_plane": "xy", "unknown_option": None},
])
def test_get_encoder_raw(encoder_options):
    info = {
        "data_type": "float32",
        "num_channels": 1,
        "scales": [
            {
                "encoding": "raw"
            }
        ]
    }
    encoder = get_encoder(info, info["scales"][0], encoder_options)
    assert isinstance(encoder, RawChunkEncoder)


@pytest.mark.parametrize("encoder_options", [
    {},
    {"jpeg_quality": 1, "jpeg_plane": "xy", "unknown_option": None},
])
def test_get_encoder_compressed_segmentation(encoder_options):
    info = {
        "data_type": "uint32",
        "num_channels": 1,
        "scales": [
            {
                "encoding": "compressed_segmentation",
                "compressed_segmentation_block_size": [8, 8, 8]
            }
        ]
    }
    encoder = get_encoder(info, info["scales"][0], encoder_options)
    assert isinstance(encoder, CompressedSegmentationEncoder)


def test_get_encoder_cseg_incomplete_info():
    info = {
        "data_type": "uint32",
        "num_channels": 1,
        "scales": [
            {
                "encoding": "compressed_segmentation"
            }
        ]
    }
    with pytest.raises(InvalidInfoError):
        get_encoder(info, info["scales"][0])


@pytest.mark.parametrize("encoder_options", [
    {},
    {"jpeg_quality": 1, "jpeg_plane": "xy", "unknown_option": None},
])
def test_get_encoder_jpeg(encoder_options):
    info = {
        "data_type": "uint8",
        "num_channels": 1,
        "scales": [
            {
                "encoding": "jpeg"
            }
        ]
    }
    encoder = get_encoder(info, info["scales"][0], encoder_options)
    assert isinstance(encoder, JpegChunkEncoder)


def test_get_encoder_invalid_info():
    info = {
        "data_type": "invalid_dtype",
        "num_channels": 1,
        "scales": [
            {
                "encoding": "raw"
            }
        ]
    }
    with pytest.raises(InvalidInfoError):
        get_encoder(info, info["scales"][0])
    info = {
        "data_type": "uint8",
        "num_channels": -1,
        "scales": [
            {
                "encoding": "raw"
            }
        ]
    }
    with pytest.raises(InvalidInfoError):
        get_encoder(info, info["scales"][0])
    info = {
        "data_type": "uint8",
        "num_channels": 1,
        "scales": [
            {
                "encoding": "invalid_encoding"
            }
        ]
    }
    with pytest.raises(InvalidInfoError):
        get_encoder(info, info["scales"][0])
    with pytest.raises(InvalidInfoError):
        get_encoder(info, {})
    with pytest.raises(InvalidInfoError):
        get_encoder({}, {})


def test_get_encoder_incompatible_dtype():
    info = {
        "data_type": "uint8",
        "num_channels": 4,
        "scales": [
            {
                "encoding": "jpeg"
            }
        ]
    }
    with pytest.raises(InvalidInfoError):
        get_encoder(info, info["scales"][0])
    info = {
        "data_type": "uint16",
        "num_channels": 1,
        "scales": [
            {
                "encoding": "compressed_segmentation",
                "compressed_segmentation_block_size": [8, 8, 8]
            }
        ]
    }
    with pytest.raises(InvalidInfoError):
        get_encoder(info, info["scales"][0])


def test_add_argparse_options():
    import argparse
    parser = argparse.ArgumentParser()
    add_argparse_options(parser)
    parser.parse_args([])

    parser = argparse.ArgumentParser()
    add_argparse_options(parser, allow_lossy=True)
    args = parser.parse_args(["--jpeg-quality", "50",
                              "--jpeg-plane", "xz"])
    assert args.jpeg_quality == 50
    assert args.jpeg_plane == "xz"


def test_raw_encoder_roundtrip():
    encoder = RawChunkEncoder("float32", 2)
    test_chunk = np.arange(11 * 50 * 64 * 2, dtype="<f").reshape(2, 64, 50, 11)
    buf = encoder.encode(test_chunk)
    decoded_chunk = encoder.decode(buf, (11, 50, 64))
    assert np.array_equal(decoded_chunk, test_chunk)


def test_raw_encoder_unsafe_cast():
    encoder = RawChunkEncoder("uint8", 2)
    test_chunk = np.ones((1, 1, 1, 1), dtype="uint16")
    with pytest.raises(Exception):
        encoder.encode(test_chunk)


def test_raw_encoder_invalid_size():
    encoder = RawChunkEncoder("uint8", 1)
    test_chunk = np.zeros((1, 1, 1, 11), dtype="uint8")
    buf = encoder.encode(test_chunk)
    with pytest.raises(InvalidFormatError):
        encoder.decode(buf, (12, 1, 1))
    decoder = RawChunkEncoder("uint16", 1)
    with pytest.raises(InvalidFormatError):
        encoder.decode(buf, (6, 1, 1))


def test_compressed_segmentation_invalid_data_type():
    with pytest.raises(IncompatibleEncoderError):
        CompressedSegmentationEncoder("uint8", 1, [8, 8, 8])


def test_compressed_segmentation_0bit_roundtrip():
    encoder = CompressedSegmentationEncoder("uint32", 1, [8, 8, 8])
    test_chunk = np.zeros((1, 2, 3, 1), dtype="<I") + 37
    buf = encoder.encode(test_chunk)
    decoded_chunk = encoder.decode(buf, (1, 3, 2))
    assert np.array_equal(decoded_chunk, test_chunk)


def test_compressed_segmentation_1bit_roundtrip():
    encoder = CompressedSegmentationEncoder("uint32", 1, [8, 8, 8])
    test_chunk = np.arange(1 * 5 * 8, dtype="<I").reshape(1, 8, 5, 1) % 2 + 37
    buf = encoder.encode(test_chunk)
    decoded_chunk = encoder.decode(buf, (1, 5, 8))
    assert np.array_equal(decoded_chunk, test_chunk)


def test_compressed_segmentation_2bit_roundtrip():
    encoder = CompressedSegmentationEncoder("uint32", 1, [8, 8, 8])
    test_chunk = np.arange(1 * 5 * 8, dtype="<I").reshape(1, 8, 5, 1) % 4 + 37
    buf = encoder.encode(test_chunk)
    decoded_chunk = encoder.decode(buf, (1, 5, 8))
    assert np.array_equal(decoded_chunk, test_chunk)


def test_compressed_segmentation_4bit_roundtrip():
    encoder = CompressedSegmentationEncoder("uint32", 1, [8, 8, 8])
    test_chunk = np.arange(1 * 5 * 8, dtype="<I").reshape(1, 8, 5, 1) % 16 + 7
    buf = encoder.encode(test_chunk)
    decoded_chunk = encoder.decode(buf, (1, 5, 8))
    assert np.array_equal(decoded_chunk, test_chunk)


def test_compressed_segmentation_8bit_roundtrip():
    encoder = CompressedSegmentationEncoder("uint32", 1, [8, 8, 8])
    test_chunk = (np.arange(14 * 5 * 8, dtype="<I").reshape(1, 8, 5, 14) % 256
                  + 452435)
    buf = encoder.encode(test_chunk)
    decoded_chunk = encoder.decode(buf, (14, 5, 8))
    assert np.array_equal(decoded_chunk, test_chunk)


# The different block sizes test 16-bit and 32-bit encodings
@pytest.mark.parametrize("block_size", [
    [8, 8, 8],
    [16, 16, 16],
    [32, 32, 32],
    [64, 64, 64],
])
def test_compressed_segmentation_roundtrip_uint32(block_size):
    encoder = CompressedSegmentationEncoder("uint32", 1, block_size)
    test_chunk = (np.arange(11 * 50 * 64, dtype="<I").reshape(1, 64, 50, 11)
                  + 75621928)
    buf = encoder.encode(test_chunk)
    decoded_chunk = encoder.decode(buf, (11, 50, 64))
    assert np.array_equal(decoded_chunk, test_chunk)


# The different block sizes test 16-bit and 32-bit encodings
@pytest.mark.parametrize("block_size", [
    [8, 8, 8],
    [16, 16, 16],
    [32, 32, 32],
    [64, 64, 64]
])
def test_compressed_segmentation_roundtrip_uint64(block_size):
    encoder = CompressedSegmentationEncoder("uint64", 1, block_size)
    test_chunk = (np.arange(64 * 64 * 63, dtype="<Q").reshape(1, 64, 64, 63)
                  + 560328569340672)
    buf = encoder.encode(test_chunk)
    decoded_chunk = encoder.decode(buf, (63, 64, 64))
    assert np.array_equal(decoded_chunk, test_chunk)


def test_cseg_decoder_invalid_data():
    encoder = CompressedSegmentationEncoder("uint32", 1, [8, 8, 8])
    with pytest.raises(InvalidFormatError):
        encoder.decode(b"", (1, 1, 1))

    test_chunk = np.ones((1, 1, 1, 11), dtype="uint8")
    buf = encoder.encode(test_chunk)
    encoder64 = CompressedSegmentationEncoder("uint64", 1, [8, 8, 8])
    test_chunk = np.ones((1, 1, 1, 11), dtype="uint8")
    with pytest.raises(InvalidFormatError):
        encoder64.decode(buf, (1, 1, 1))


def test_jpeg_invalid_data_type():
    with pytest.raises(IncompatibleEncoderError):
        JpegChunkEncoder("uint16", 1, 95, "xy")


def test_jpeg_invalid_num_channels():
    with pytest.raises(IncompatibleEncoderError):
        JpegChunkEncoder("uint8", 4, 95, "xy")


@pytest.mark.parametrize("plane", ["xy", "xz"])
def test_jpeg_roundtrip_greyscale(plane):
    encoder = JpegChunkEncoder("uint8", 1, 95, plane)
    n = np.newaxis
    test_chunk = (np.linspace(0, 100, 50, dtype="B")[n, :, n, n]
                  + np.linspace(50, 0, 11, dtype="B")[n, n, :, n]
                  + np.linspace(0, 155, 64, dtype="B")[n, n, n, :])
    buf = encoder.encode(test_chunk)
    decoded_chunk = encoder.decode(buf, (64, 11, 50))
    assert decoded_chunk.shape == test_chunk.shape
    assert (np.mean(np.square(decoded_chunk.astype("f") - test_chunk))
            < 0.51 * 1.1)


@pytest.mark.parametrize("plane", ["xy", "xz"])
def test_jpeg_roundtrip_rgb(plane):
    encoder = JpegChunkEncoder("uint8", 3, 95, plane)
    n = np.newaxis
    test_chunk = np.stack(np.broadcast_arrays(
        np.linspace(0, 200, 50, dtype="B")[:, n, n],
        np.linspace(255, 100, 11, dtype="B")[n, :, n],
        np.linspace(20, 200, 64, dtype="B")[n, n, :]))
    buf = encoder.encode(test_chunk)
    decoded_chunk = encoder.decode(buf, (64, 11, 50))
    assert decoded_chunk.shape == test_chunk.shape
    assert (np.mean(np.square(decoded_chunk.astype("f") - test_chunk))
            < 1.36 * 1.1)


def test_jpeg_decoder_invalid_data():
    encoder = JpegChunkEncoder("uint8", 1)
    with pytest.raises(InvalidFormatError):
        encoder.decode(b"", (1, 1, 1))

    encoder_3ch = JpegChunkEncoder("uint8", 3)
    test_chunk = np.ones((1, 1, 1, 11), dtype="uint8")
    buf = encoder.encode(test_chunk)
    with pytest.raises(InvalidFormatError):
        encoder.decode(buf, (2, 1, 1))
    with pytest.raises(InvalidFormatError):
        encoder_3ch.decode(buf, (1, 1, 1))
