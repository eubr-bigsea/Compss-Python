#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__ = "Lucas Miguel S Ponce"
__email__ = "lucasmsp@gmail.com"

from pycompss.api.task import task
import pandas as pd
import numpy as np


class IntersectionOperation(object):

    @staticmethod
    def transform(data1, data2):
        """
        Returns a new DataFrame containing rows in both frames.

        :param data1: A list with nfrag pandas's dataframe;
        :param data2: Other list with nfrag pandas's dataframe;
        :return: Returns a new pandas dataframe

        .. note:: Rows with NA elements will not be take in count.
        """
        result = data1[:]

        nfrag1 = len(data1)
        nfrag2 = len(data2)

        for f1 in xrange(nfrag1):
            for f2 in xrange(nfrag2):
                last = (f2 == (nfrag2-1))
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
            # if is the first iteration, remove all rows with NA values.
            df1 = df1.dropna(axis=0, how='any')

        df2 = df2.dropna(axis=0, how='any')
        keys = df1.columns.tolist()
        df1 = pd.merge(df1, df2, how='left', on=keys,
                       indicator=True, copy=False)

        def combine_indicators(col1, col2):
            """Combine indicators of _merge column."""
            if ('b' in col1) or ('b' in col2):
                return 'both'
            else:
                return 'left_only'

        if index > 0:
            df1['_merge'] = \
                np.vectorize(combine_indicators)(col1=df1['_merge'],
                                                 col2=indicator)

        if last:
            df1 = df1.loc[df1['_merge'] == 'both', keys]
    return df1



