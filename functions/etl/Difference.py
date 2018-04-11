#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__ = "Lucas Miguel S Ponce"
__email__ = "lucasmsp@gmail.com"

from pycompss.api.task import task
from pycompss.api.parameter import *
import pandas as pd


class DifferenceOperation(object):
    """Difference Operation.

    Returns a new set with containing rows in the first frame but not
    in the second one.
    """

    def transform(self, data1, data2, numFrag):
        """DifferenceOperation.

        :param data1: A list with numFrag pandas's dataframe;
        :param data2: The second list with numFrag pandas's dataframe.
        :return: A list with numFrag pandas's dataframe.
        """
        if all([len(data1) != numFrag, len(data2) != numFrag]):
            raise Exception("data1 and data2 must have len equal to numFrag.")

        result = data1[:]
        for f1 in range(numFrag):
            for f2 in range(numFrag):
                result[f1] = self._difference(result[f1], data2[f2])

        return result

    @task(isModifier=False, returns=list)
    def _difference(self, df1, df2):
        """Peform a Difference partial operation."""
        if len(df1) > 0:
            if len(df2) > 0:
                names = df1.columns
                df1 = pd.merge(df1, df2, indicator=True, how='left', on=None)
                df1 = df1.loc[df1['_merge'] == 'left_only', names]
        return df1
