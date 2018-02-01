# Copyright (c) 2018 Forschungszentrum Juelich GmbH
# Author: Yann Leprince <y.leprince@fz-juelich.de>
#
# This software is made available under the MIT licence, see LICENCE.txt.

def ceil_div(a, b):
    """Ceil integer division (math.ceil(a / b) using integer arithmetic)"""
    return (a - 1) // b + 1
