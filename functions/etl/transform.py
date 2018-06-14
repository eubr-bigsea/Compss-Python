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

    def transform(self, data, settings, nfrag):
        """TransformOperation.

        :param data:  A list with nfrag pandas's dataframe;
        :settings: A dictionary that contains:
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
        functions = self.preprocessing(settings)
        result = [[] for _ in range(nfrag)]
        for f in range(nfrag):
            result[f] = _apply(data[f], functions)
        return result

    def preprocessing(self, settings):
        """Check all the settings."""
        functions = settings.get('functions', [])
        if any([len(functions) == 0,
                any([True if (len(f) != 3) else False for f in functions])
                ]):
            raise Exception('You must inform a valid `functions` parameter.')
        return functions

    def transform_serial(self, data, functions):
        """Apply the Transformation operation in each row."""
        return _apply_(data, functions)


@task(returns=list)
def _apply(data, functions):
    """Apply the Transformation operation in each row."""
    return _apply_(data, functions)


def _apply_(data, functions):
    """Apply the Transformation operation in each row."""
    for action in functions:
        ncol, function, imp = action
        exec(imp)
        if len(data) > 0:
            func = eval(function)
            v1s = []
            for _, row in data.iterrows():
                v1s.append(func(row))
            data[ncol] = v1s
        else:
            data[ncol] = np.nan
    return data


def group_datetime(d, interval):
    """Group datetime in bins."""
    seconds = d.second + d.hour*3600 + d.minute*60 + d.microsecond/1000
    k = d - datetime.timedelta(seconds=seconds % interval)
    return datetime.datetime(k.year, k.month, k.day,
                             k.hour, k.minute, k.second)
