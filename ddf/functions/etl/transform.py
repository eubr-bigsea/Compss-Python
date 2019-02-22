#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__ = "Lucas Miguel S Ponce"
__email__ = "lucasmsp@gmail.com"

import numpy as np
import pandas as pd

# basic imports:
import datetime
import time
from dateutil.parser import parse


"""

"""

#TODO: check apply method
class TransformOperation(object):
    """
    Returns a new DataFrame applying the expression to the specified column.
    """

    def transform(self, data, settings):
        """
        :param data:  A list with nfrag pandas's dataframe;
        :param settings: A dictionary that contains:
            - function: A lambda function;
            - alias: New column name;
        :return: Returns a list with nfrag pandas's dataframe with
                   the news columns.
        """

        return _apply(data, settings)


def _apply(df, settings):
    """Apply the Transformation operation in each row."""

    """
    see also: 
    https://engineering.upside.com/a-beginners-guide-to-optimizing-pandas-
    code-for-speed-c09ef2c6a4d6
    """
    function = settings['function']
    new_column = settings['alias']

    if len(df) > 0:
        # vectorized_function = np.vectorize(function)
        df[new_column] = df.apply(function, axis=1)

    else:
        df[new_column] = np.nan
    return df


def group_datetime(d, interval):
    """Group datetime in bins."""
    seconds = d.second + d.hour*3600 + d.minute*60 + d.microsecond/1000
    k = d - datetime.timedelta(seconds=seconds % interval)
    return datetime.datetime(k.year, k.month, k.day,
                             k.hour, k.minute, k.second)
