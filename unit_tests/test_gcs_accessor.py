from urllib.parse import quote_plus

import pytest
from neuroglancer_scripts.gcs_accessor import GoogleBucketParser, UrlParser


@pytest.fixture
def gs_parser():
    yield UrlParser("gs://foo/bar")


def test_url_parser_init_type(gs_parser):
    assert isinstance(gs_parser, GoogleBucketParser)


def test_gs_url_parser_base_url(gs_parser):
    assert gs_parser.base_url == "foo/bar"


url_format_args = [
    ([], None, TypeError),
    (
        ["buzz.txt"],
        "https://www.googleapis.com/storage/v1/b/{bucketname}/o/{final_path}?alt=media".format(
            bucketname="foo", final_path=quote_plus("bar/buzz.txt")
        ),
        None,
    ),
]


@pytest.mark.parametrize("paths, expected, err", url_format_args)
def test_gs_url_parser_format_url(paths, expected, err, gs_parser):
    if err is not None:
        with pytest.raises(err):
            gs_parser.format_url(*paths)
        return
    assert expected == gs_parser.format_url(*paths)
