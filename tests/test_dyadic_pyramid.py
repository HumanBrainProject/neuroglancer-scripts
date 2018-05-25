# Copyright (c) 2018 CEA
# Author: Yann Leprince <yann.leprince@cea.fr>
#
# This software is made available under the MIT licence, see LICENCE.txt.

from neuroglancer_scripts.dyadic_pyramid import (
    choose_unit_for_key,
)


def test_choose_unit_for_key():
    assert choose_unit_for_key(1e-3) == "pm"
    assert choose_unit_for_key(1) == "nm"
    assert choose_unit_for_key(1e3) == "um"
    assert choose_unit_for_key(1e6) == "mm"
    assert choose_unit_for_key(1e9) == "m"
