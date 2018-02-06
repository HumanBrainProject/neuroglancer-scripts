# Copyright (c) 2018 Forschungszentrum Juelich GmbH
# Author: Yann Leprince <y.leprince@fz-juelich.de>
#
# This software is made available under the MIT licence, see LICENCE.txt.

import numpy as np
import pytest

from neuroglancer_scripts.utils import *


def test_ceil_div():
    assert ceil_div(0, 8) == 0
    assert ceil_div(1, 8) == 1
    assert ceil_div(7, 8) == 1
    assert ceil_div(8, 8) == 1
    assert ceil_div(9, 8) == 2
    with pytest.raises(ZeroDivisionError):
        ceil_div(1, 0)


def test_permute():
    assert permute((1, 2, 3), (0, 1, 2)) == (1, 2, 3)
    assert permute((1, 2, 3), (2, 0, 1)) == (3, 1, 2)


def test_invert_permutation():
    assert np.array_equal(invert_permutation((0, 1, 2)), [0, 1, 2])
    assert np.array_equal(invert_permutation((2, 1, 0)), [2, 1, 0])
    assert np.array_equal(invert_permutation((2, 0, 1)), [1, 2, 0])


def test_readable_count():
    assert readable_count(0) == "0 "
    assert readable_count(1) == "1 "
    assert readable_count(512) == "512 "
    assert readable_count(1e10) == "9.3 Gi"
    # Test fall-back for the largest unit
    assert readable_count(2 ** 70) == "1,024 Ei"
