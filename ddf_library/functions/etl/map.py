#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Lucas Miguel S Ponce"
__email__ = "lucasmsp@gmail.com"

from ddf_library.utils import generate_info
import numpy as np
import datetime


def map(data, settings):
    """
    Returns a new DataFrame applying the expression to the specified column.

    :param data:  A pandas's DataFrame;
    :param settings: A dictionary that contains:
        - function: A lambda function;
        - alias: New column name;
    :return: Returns pandas's DataFrame with the news columns.

    .. seealso::
    https://engineering.upside.com/a-beginners-guide-to-optimizing-pandas-
    code-for-speed-c09ef2c6a4d6
    """
    function = settings['function']
    new_column = settings['alias']
    frag = settings['id_frag']

    if len(data) > 0:
        data[new_column] = data.apply(function, axis=1)

    else:
        data[new_column] = np.nan

    info = generate_info(data, frag)
    return data, info


def group_datetime(d, interval):
    """Group datetime in bins."""
    seconds = d.second + d.hour*3600 + d.minute*60 + d.microsecond/1000
    k = d - datetime.timedelta(seconds=seconds % interval)
    return datetime.datetime(k.year, k.month, k.day,
                             k.hour, k.minute, k.second)
