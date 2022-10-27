#  Copyright (c) 2022. RISC Software GmbH.
#  All rights reserved.

import sys
from typing import Optional, Any

import numpy as np


def headline(text, block_width=50):
    print('-' * block_width)
    print('--- {} '.format(text).ljust(block_width, '-'))
    print('-' * block_width)


def native(val, dtype=None, force=False):
    """
    Converts value from a numpy to native python type if applicable, otherwise pass through.
    :param val: value to convert
    :param dtype: optional dtype to cast to
    :param force: convert also if val is not a numpy type
    :return: 'native' python value
    """
    if hasattr(val, 'dtype'):
        if np.isscalar(val):
            return val.item() if dtype is None else val.astype(dtype).item()
        return np.asarray(val, dtype=dtype).tolist()
    elif force:
        return native(np.asarray(val, dtype=dtype))
    return val


def ndtuple(val, d=3):
    """
    Converts the value to a nd tuple if it is a scalar, otherwise pass it through.
    Note: the method does not enforce d dimensions on non-scalar types.
    :param val: value to convert
    :param d: dimensionality of the output value
    :return: n-dimensional value
    """
    if np.isscalar(val):
        val = (val, ) * d
    return tuple(val)


def as_tuple(a):
    """
    Convert an item which may be a container or a scalar to a tuple
    """
    if hasattr(a, '__iter__') and not isinstance(a, str):
        return tuple(a)
    return (a, ) if a is not None else tuple()


def default(val: Optional[Any], d: Any):
    """
    returns a default value if val is not set (None) or otherwise val
    :param val: value to check
    :param d: default value
    """
    return d if val is None else val


def warn(msg):
    print(msg, file=sys.stderr)
    sys.stderr.flush()
    sys.stdout.flush()


def nan_clip(v, lower=0.0, upper=1.0):
    return v if lower <= v <= upper else np.nan


def split_prefix_number(val: str):
    idx = next((idx for idx, c in enumerate(val) if c.isdigit()), None)
    if idx is None:
        raise RuntimeError("No numeric part in '{}'".format(val))
    prefix = val[:idx]
    number = val[idx:]
    if not number.isdigit():
        raise RuntimeError("Numeric part of '{}' contains non-digits".format(val))

    return prefix, int(number)


