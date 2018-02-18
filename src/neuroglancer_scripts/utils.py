# Copyright (c) 2018 Forschungszentrum Juelich GmbH
# Author: Yann Leprince <y.leprince@fz-juelich.de>
#
# This software is made available under the MIT licence, see LICENCE.txt.


"""Miscellaneous utility functions.
"""

import collections

import numpy as np


__all__ = [
    "ceil_div",
    "permute",
    "invert_permutation",
    "readable_count",
    "LENGTH_UNITS",
    "format_length",
]


def ceil_div(a, b):
    """Ceil integer division (``ceil(a / b)`` using integer arithmetic)."""
    return (a - 1) // b + 1


def permute(seq, p):
    """Permute the elements of a sequence according to a permutation.

    :param seq: a sequence to be permuted
    :param p: a permutation (sequence of integers between ``0`` and
              ``len(seq) - 1``)
    :returns: ``tuple(seq[i] for i in p)``
    :rtype: tuple
    """
    return tuple(seq[i] for i in p)


def invert_permutation(p):
    """Invert a permutation.

    :param p: a permutation (sequence of integers between ``0`` and
              ``len(p) - 1``)
    :returns: an array ``s``, where ``s[i]`` gives the index of ``i`` in ``p``
    :rtype: numpy.ndarray
    """
    p = np.asarray(p)
    s = np.empty(p.size, p.dtype)
    s[p] = np.arange(p.size)
    return s


_IEC_PREFIXES = [
    (2 ** 10, "ki"),
    (2 ** 20, "Mi"),
    (2 ** 30, "Gi"),
    (2 ** 40, "Ti"),
    (2 ** 50, "Pi"),
    (2 ** 60, "Ei"),
]


def readable_count(count):
    """Format a number to a human-readable string with an IEC binary prefix.

    The resulting string has a minimum of 2 significant digits. It is never
    longer than 6 characters, unless the input exceeds 2**60. You are expected
    to concatenate the result with the relevant unit (e.g. B for bytes):

    >>> readable_count(512) + "B"
    '512 B'
    >>> readable_count(1e10) + "B"
    '9.3 GiB'

    :param int count: number to be converted (must be >= 0)
    :returns: a string representation of the number with an IEC binary prefix
    :rtype: str

    """
    assert count >= 0
    num_str = format(count, ".0f")
    if len(num_str) <= 3:
        return num_str + " "
    for factor, prefix in _IEC_PREFIXES:
        if count > 10 * factor:
            num_str = format(count / factor, ".0f")
        else:
            num_str = format(count / factor, ".1f")
        if len(num_str) <= 3:
            return num_str + " " + prefix
    # Fallback: use the last prefix
    factor, prefix = _IEC_PREFIXES[-1]
    return "{:,.0f} {}".format(count / factor, prefix)


LENGTH_UNITS = collections.OrderedDict([
    ("km", 1e-12),
    ("m", 1e-9),
    ("mm", 1e-6),
    ("um", 1e-3),
    ("nm", 1.),
    ("pm", 1e3),
])
"""List of physical units of length."""


def format_length(length_nm, unit):
    """Format a length according to the provided unit (input in nanometres).

    :param float length_nm: a length in nanometres
    :param str unit: must be one of ``LENGTH_UNITS.keys``
    :return: the formatted length, rounded to the specified unit (no fractional
             part is printed)
    :rtype: str
    """
    return format(length_nm * LENGTH_UNITS[unit], ".0f") + unit
