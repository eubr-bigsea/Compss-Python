#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__ = "Lucas Miguel S Ponce"
__email__ = "lucasmsp@gmail.com"

from pycompss.api.task import task
from pycompss.api.parameter import *
import numpy as np
import pandas as pd

# basic imports:
import datetime
import time
from dateutil.parser import parse


class TransformOperation(object):
    """Transform Operation.

    Returns a new DataFrame applying the expression to the specified column.
    """

    def transform(self, data, settings, numFrag):
        """TransformOperation.

        :param data:      A list with numFrag pandas's dataframe;
        :settings:        A dictionary that contains:
        - functions:   A list with an array with 3-dimensions.
          * 1ª position:  The lambda function to be applied as a string;
          * 2ª position:  The alias to new column to be applied the function;
          * 3ª position:  The string to import some needed module
                          ('' if isnt needed);
        :return:   Returns a list with numFrag pandas's dataframe with
                   the news columns.

        ex.:
        settings['functions'] = [['alias_col1',
                                 "lambda col: np.add(col['col1'],col['col2'])",
                                '']]
        """
        functions = self.validate(settings)
        result = [[] for f in range(numFrag)]
        for f in range(numFrag):
            result[f] = self._apply(data[f], functions)
        return result

    def validate(self, settings):
        """Check all the settings."""
        functions = settings.get('functions', [])
        if any([len(functions) == 0,
                any([True if (len(f) != 3) else False for f in functions])
                ]):
            raise Exception('You must inform a valid `functions` parameter.')
        return functions

    @task(isModifier=False, returns=list)
    def _apply(self, data, functions):
        """Apply the Transformation operation in each row."""
        for function in functions:
            ncol, function, imp = function
            exec(imp)
            if len(data) > 0:
                func = eval(function)
                v1s = []
                for _, row in data.iterrows():
                    v1 = func(row)
                    v1s.append(v1)
                data[ncol] = v1s
            else:
                data[ncol] = np.nan
        return data

    def apply_serial(self, data, functions):
        """Apply the Transformation operation in each row."""
        for function in functions:
            ncol, function, imp = function
            exec(imp)
            if len(data) > 0:
                func = eval(function)
                v1s = []
                for _, row in data.iterrows():
                    v1 = func(row)
                    v1s.append(v1)
                data[ncol] = v1s
            else:
                data[ncol] = np.nan
        return data

    def group_datetime(self, d, interval):
        """Group datetime in bins."""
        seconds = d.second + d.hour*3600 + d.minute*60 + d.microsecond/1000
        k = d - datetime.timedelta(seconds=seconds % interval)
        return datetime.datetime(k.year, k.month, k.day,
                                 k.hour, k.minute, k.second)
