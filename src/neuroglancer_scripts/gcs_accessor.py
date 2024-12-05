from abc import ABC
from typing import Dict
from urllib.parse import quote_plus

import neuroglancer_scripts.http_accessor


# TODO move to its own module?
class UrlParser(ABC):
    base_url: str = None

    def __new__(cls, path: str):
        for registered_protocol in cls._PROTOCOL_DICT:
            if path.startswith(f"{registered_protocol}//"):
                path = path.replace(f"{registered_protocol}//", "")
                use_cls = cls._PROTOCOL_DICT[registered_protocol]
                inst: UrlParser = object.__new__(use_cls)
                inst.base_url = path
                return inst
        return super().__new__(cls)

    protocol: str = None
    _PROTOCOL_DICT: Dict[str, "UrlParser"] = {}

    def __post_init__(self):
        assert self.base_url is not None, "UrlParser.base_url must be defined!"

    def __init_subclass__(cls, protocol: str):

        if protocol in cls._PROTOCOL_DICT:
            print(f"{protocol} already registered. Overriding")
        cls._PROTOCOL_DICT[protocol] = cls
        cls.protocol = protocol
        return super().__init_subclass__()

    def format_url(self, path: str):
        raise NotImplementedError("Child class must override format_url")


class GoogleBucketParser(UrlParser, protocol="gs:"):

    def format_url(self, path: str):
        bucketname, prefix = self.base_url.split("/", maxsplit=1)
        final_path = quote_plus(f"{prefix}/{path}")
        return f"https://www.googleapis.com/storage/v1/b/{bucketname}/o/{final_path}?alt=media"


class GCSAccessor(neuroglancer_scripts.http_accessor.HttpAccessor):
    can_read = True
    can_write = False

    def __init__(self, base_url):
        super().__init__(base_url)
        self.parser = GoogleBucketParser(f"gs://{base_url}")
        self.parser.base_url = base_url

    def format_path(self, relative_path):
        return self.parser.format_url(relative_path)

    def fetch_file(self, relative_path):
        import json

        import numpy as np

        if relative_path == "transform.json":
            return json.dumps(np.eye(4).tolist()).encode()
        return super().fetch_file(relative_path)
