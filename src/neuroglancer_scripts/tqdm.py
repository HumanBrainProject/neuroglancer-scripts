# Copyright (c) 2018, CEA
# Author: Yann Leprince <yann.leprince@cea.fr>
#
# This software is made available under the MIT licence, see LICENCE.txt.

"""Replacement functions for making tqdm optional.

Only a few basic usages are supported on the replacement objects. Internal use
only.
"""

import os


__all__ = [
    'tqdm',
    'trange',
]


tqdm = None
if 'NO_TQDM' not in os.environ:
    try:
        import tqdm as real_tqdm
    except ImportError:
        pass


class FakeTqdm:
    def __new__(cls, iterable=None, **kwargs):
        if iterable is None:
            return object.__new__(cls)
        return iterable

    def update(self):
        pass

    @staticmethod
    def write(*args, **kwargs):
        print(*args, **kwargs)


def fake_trange(*args, **kwargs):
    yield from range(*args)


if tqdm is None:
    tqdm = FakeTqdm
    trange = fake_trange
else:
    tqdm = real_tqdm.tqdm
    trange = real_tqdm.trange
