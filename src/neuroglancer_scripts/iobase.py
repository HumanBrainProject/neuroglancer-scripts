from abc import ABC, abstractmethod


class MultiResIOBase(ABC):

    @property
    @abstractmethod
    def info(self):
        raise NotImplementedError

    @abstractmethod
    def iter_scale(self):
        raise NotImplementedError

    @abstractmethod
    def scale_info(self, scale_key):
        raise NotImplementedError

    @abstractmethod
    def read_chunk(self, scale_key, chunk_coords):
        raise NotImplementedError

    @abstractmethod
    def write_chunk(self, chunk, scale_key, chunk_coords):
        raise NotImplementedError
