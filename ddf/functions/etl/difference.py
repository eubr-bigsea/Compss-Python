#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__ = "Lucas Miguel S Ponce"
__email__ = "lucasmsp@gmail.com"

from pycompss.api.task import task
import pandas as pd


class DifferenceOperation(object):
    """Difference Operation.

    Returns a new set with containing rows in the first frame but not
    in the second one.

    Optimization: No
    """
    def transform(self, data1, data2, nfrag):
        """DifferenceOperation.

        :param data1: A list with nfrag pandas's dataframe;
        :param data2: The second list with nfrag pandas's dataframe;
        :param nfrag: The number of fragments;
        :return: A list with nfrag pandas's dataframe.

        !LEMONADE_NOTE: create a different function only to the last iteration
        """
        self.preprocessing(data1, data2, nfrag)
        result = data1[:]
        for f1 in range(nfrag):
            for f2 in range(nfrag):
                result[f1] = _difference(result[f1], data2[f2])

        return result
    
    def preprocessing(self, data1, data2, nfrag):
        if all([len(data1) != nfrag, len(data2) != nfrag]):
            raise Exception("data1 and data2 must have len equal to nfrag.")


@task(returns=list)
def _difference(df1, df2):
    """Peform a Difference partial operation."""
    if len(df1) > 0:
        if len(df2) > 0:
            names = df1.columns
            df1 = pd.merge(df1, df2, indicator=True, how='left', on=None)
            df1 = df1.loc[df1['_merge'] == 'left_only', names]
    return df1



