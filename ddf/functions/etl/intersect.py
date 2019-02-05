#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__ = "Lucas Miguel S Ponce"
__email__ = "lucasmsp@gmail.com"

from pycompss.api.task import task
import pandas as pd
import numpy as np


#! TODO: Validate

class IntersectionOperation(object):
    """Set-Intersect.

    Returns a new DataFrame containing rows in both frames.
    """

    def transform(self, data1, data2, nfrag):
        """IntersectionOperation.

        :param data1:  A list with nfrag pandas's dataframe;
        :param data2:  Other list with nfrag pandas's dataframe;
        :param nfrag: The number of fragments;
        :return:       Returns a new pandas dataframe

        Note: rows with NA elements will not be take in count.
        """
        result = data1[:]

        for f1 in xrange(nfrag):
            for f2 in xrange(nfrag):
                last = (f2 == (nfrag-1))
                result[f1] = _intersection(result[f1], data2[f2],
                                           f2, last)

        return result


@task(returns=list)
def _intersection(df1, df2, index, last):
    """Perform a partial intersection."""
    if len(df1) > 0:
        if index > 0:
            indicator = df1['_merge'].values
            df1.drop(['_merge'], axis=1, inplace=True)
        else:
            df1 = df1.dropna(axis=0, how='any')
        df2 = df2.dropna(axis=0, how='any')
        keys = df1.columns.tolist()
        df1 = pd.merge(df1, df2, how='left', on=keys,
                       indicator=True, copy=False)
        if index > 0:
            df1['_merge'] = \
                np.vectorize(combine_indicators)(col1=df1['_merge'],
                                                 col2=indicator)

        if last:
            df1 = df1.loc[df1['_merge'] == 'both', keys]
    return df1


def combine_indicators(col1, col2):
    """Combine indicators of _merge column."""
    if ('b' in col1) or ('b' in col2):
        return 'both'
    else:
        return 'left_only'
