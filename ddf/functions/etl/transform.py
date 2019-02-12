#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__ = "Lucas Miguel S Ponce"
__email__ = "lucasmsp@gmail.com"

from pycompss.api.task import task
import numpy as np
import pandas as pd

# basic imports:
import datetime
import time
from dateutil.parser import parse


class TransformOperation(object):
    """Transform Operation.

    Returns a new DataFrame applying the expression to the specified column.


    #CHECK apply method
    """

    def transform(self, data, settings):
        """TransformOperation.

        :param data:  A list with nfrag pandas's dataframe;
        :param settings: A dictionary that contains:
        - functions: A list with an array with 3-dimensions.
          * 1ª position: The lambda function to be applied as a string;
          * 2ª position: The alias to new column to be applied the function;
          * 3ª position: The string to import some needed module
                          ('' if isnt needed);
        :return: Returns a list with nfrag pandas's dataframe with
                   the news columns.

        ex.:
        settings['functions'] = [['alias_col1',
                                 "lambda col: np.add(col['col1'],col['col2'])",
                                '']]
        """

        return _apply(data, settings)


def _apply(df, settings):
    """Apply the Transformation operation in each row."""

    function = settings['function']
    new_column = settings['alias']

    if len(df) > 0:
        v1s = []
        for _, row in df.iterrows():
            v1s.append(function(row))
        df[new_column] = v1s
    else:
        df[new_column] = np.nan
    return df


def group_datetime(d, interval):
    """Group datetime in bins."""
    seconds = d.second + d.hour*3600 + d.minute*60 + d.microsecond/1000
    k = d - datetime.timedelta(seconds=seconds % interval)
    return datetime.datetime(k.year, k.month, k.day,
                             k.hour, k.minute, k.second)
