#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__ = "Lucas Miguel S Ponce"
__email__ = "lucasmsp@gmail.com"

from pycompss.api.task import task
from pycompss.api.parameter import *
import pandas as pd
import numpy as np


class IntersectionOperation(object):
    """Set-Intersect.

    Returns a new DataFrame containing rows in both frames.
    """

    def transform(self, data1, data2, numFrag):
        """IntersectionOperation.

        :param data1:  A list with numFrag pandas's dataframe;
        :param data2:  Other list with numFrag pandas's dataframe;
        :return:       Returns a new pandas dataframe

        Note: rows with NA elements will not be take in count.
        """
        result = data1[:]

        for f1 in xrange(numFrag):
            for f2 in xrange(numFrag):
                last = (f2 == (numFrag-1))
                result[f1] = self._intersection(result[f1], data2[f2],
                                                f2, last)

        return result

    @task(isModifier=False, returns=list)
    def _intersection(self, df1, df2, index, last):
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
                    np.vectorize(self.combine_indicators)(col1=df1['_merge'],
                                                          col2=indicator)

            if last:
                df1 = df1.loc[df1['_merge'] == 'both', keys]
        return df1

    def combine_indicators(self, col1, col2):
        """Combine indicators of _merge column."""
        if ('b' in col1) or ('b' in col2):
            return 'both'
        else:
            return 'left_only'


    # @task(returns=list)
    # def merge_intersect(self, list1, list2):
    #     """Merge partial intersections."""
    #     result = pd.concat([list1, list2], ignore_index=True)
    #     return result
