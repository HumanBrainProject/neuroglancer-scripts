# Copyright (c) 2018 Forschungszentrum Juelich GmbH
# Author: Yann Leprince <y.leprince@fz-juelich.de>
#
# This software is made available under the MIT licence, see LICENCE.txt.

import numpy as np


def ceil_div(a, b):
    """Ceil integer division (math.ceil(a / b) using integer arithmetic)"""
    return (a - 1) // b + 1


def permute(seq, p):
    """Permute the elements of seq according to the permutation p"""
    return tuple(seq[i] for i in p)


def invert_permutation(p):
    """The argument p is assumed to be some permutation of 0, 1, ..., len(p)-1.
    Returns an array s, where s[i] gives the index of i in p.
    """
    p = np.asarray(p)
    s = np.empty(p.size, p.dtype)
    s[p] = np.arange(p.size)
    return s


SI_PREFIXES = [
    (1, ""),
    (1024, "ki"),
    (1024 * 1024, "Mi"),
    (1024 * 1024 * 1024, "Gi"),
    (1024 * 1024 * 1024 * 1024, "Ti"),
    (1024 * 1024 * 1024 * 1024 * 1024, "Pi"),
    (1024 * 1024 * 1024 * 1024 * 1024 * 1024, "Ei"),
]


def readable_count(count):
    for factor, prefix in SI_PREFIXES:
        if count > 10 * factor:
            num_str = format(count / factor, ".0f")
        else:
            num_str = format(count / factor, ".1f")
        if len(num_str) <= 3:
            return num_str + " " + prefix
    # Fallback: use the last prefix
    factor, prefix = SI_PREFIXES[-1]
    return "{:,.0f} {}".format(count / factor, prefix)
