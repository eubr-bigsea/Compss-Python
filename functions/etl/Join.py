#!/usr/bin/python
# -*- coding: utf-8 -*-
"""Join Operation.

Joins with another DataFrame, using the given join expression.
"""

__author__ = "Lucas Miguel S Ponce"
__email__ = "lucasmsp@gmail.com"

from pycompss.api.parameter import *
from pycompss.api.task import task
import pandas as pd


class JoinOperation(object):

    def __init__(self):
        pass

    def transform(self, data1, data2, params, numFrag):
        """JoinOperation.

        :param data1: A list with numFrag pandas's dataframe;
        :param data2: Other list with numFrag pandas's dataframe;
        :param params: A dictionary that contains:
            - 'option': 'inner' to InnerJoin, 'left' to left join and
                        'right' to right join.
            - 'key1': A list of keys of the first dataframe;
            - 'key2': A list of keys of the second dataframe;
            - 'case': True to case-sensitive (default, True);
            - 'keep_keys': True to keep the keys of the second dataset,
                           (default, False).
            - 'suffixes': Suffixes for attributes, a list with 2 values
                          (default, [_l,_r]);
        :param numFrag: The number of fragments;
        :return: Returns a list with numFrag pandas's dataframe.
        """
        result = [[] for i in range(numFrag)]
        key1 = params.get('key1', [])
        key2 = params.get('key2', [])
        TYPE = params.get('option', 'inner')
        sort = params.get('sort', True)

        if any([len(key1) == 0,
                len(key2) == 0,
                len(key1) != len(key2),
                TYPE not in ['inner', 'left', 'right']
                ]):
            raise \
                Exception('You must inform the keys of first '
                          'and second dataframe. You also must '
                          'inform the join type (inner,left or right join).')

        if not sort:
            if TYPE == "right":
                for f1 in range(numFrag):
                    for f2 in range(numFrag):
                        last = (f2 == numFrag-1)
                        result[f1] = self._join(result[f1], data2[f1], data1[f2],
                                                params, f2, last)
            else:
                for f1 in range(numFrag):
                    for f2 in range(numFrag):
                        last = (f2 == numFrag-1)
                        result[f1] = _join(result[f1], data1[f1], data2[f2],
                                           params, f2, last)
        else:
            sortIdx1 = [[] for i in range(numFrag)]
            sortIdx2 = [[] for i in range(numFrag)]
            for i in range(numFrag):
                sortIdx1[i] = self.join_partial_sort(data1[i], key1)
                sortIdx2[i] = self.join_partial_sort(data2[i], key2)

            from pycompss.api.api import compss_wait_on
            #sortIdx1 = compss_wait_on(sortIdx1)
            #sortIdx2 = compss_wait_on(sortIdx2)
            # print sortIdx1
            # print sortIdx2

            if TYPE == "right":
                overlapping = overlap(sortIdx2, sortIdx1, numFrag)
                overlapping = compss_wait_on(overlapping)
                for f1 in range(numFrag):
                    filled = False
                    for f2 in range(numFrag):
                        over, last = overlapping[f1][f2]
                        if over:
                            result[f1] = _join(result[f1], data2[f1],
                                              data1[f2], params, f2, last)
                            filled = True
                    if not filled:
                        result[f1] = self.fix_columns(data2[f1], sortIdx1[0], params)
            else:
                overlapping = overlap(sortIdx1, sortIdx2, numFrag)
                overlapping = compss_wait_on(overlapping)
                for f1 in range(numFrag):
                    filled = False
                    for f2 in range(numFrag):
                        over, last = overlapping[f1][f2]
                        if over:
                            result[f1] = _join(result[f1], data1[f1],
                                               data2[f2], params, f2, last)
                            filled = True
                    if not filled:
                        result[f1] = self.fix_columns(data1[f1], sortIdx2[0], params)

        return result


    def rename_cols(cols1, cols2, key, suf, keep, op):
        """Rename columns based in the columns of other dataset."""
        convert = {}
        for col in cols1:
            if (col in cols2):
                if not keep and op and col in key:
                    pass
                else:
                    n_col = "{}{}".format(col, suf)
                    convert[col] = n_col
                    key = [n_col if x == col else x for x in key]

        return convert, key


    @task(returns=list)
    def join(result, data1, data2, params, index, last):
        """Peform a partial join and a concatenation with the previosly join."""
        case_sensitive = params.get('case', True)
        keep = params.get('keep_keys', False)
        suffixes = params.get('suffixes', ['_l', '_r'])
        key1 = params['key1']
        key2 = params['key2']
        LSuf = suffixes[0]
        RSuf = suffixes[1]

        if params['option'] == 'right':
            params['option'] = 'left'
            key1, key2 = key2, key1
            LSuf, RSuf = RSuf, LSuf

        cols1 = data1.columns
        cols2 = data2.columns

        # Adding the suffixes before join.
        # This is necessary to preserve the keys
        # of the second table even though with equal name.
        convert1, key1 = self.rename_cols(cols1, cols2, key1, LSuf, keep, True)
        data1 = data1.rename(columns=convert1)
        convert2, key2 = self.rename_cols(cols2, cols1, key2, RSuf, keep, False)
        data2 = data2.rename(columns=convert2)

        # Removing rows where NaN is in keys
        data1 = data1.dropna(axis=0, how='any', subset=key1)
        data2 = data2.dropna(axis=0, how='any', subset=key2)

        # import pandas.api.types as pt
        # print pt.infer_dtype(data1[key1], skipna=True)
        # print pt.infer_dtype(data2[key2], skipna=True)

        if case_sensitive:
            if not last:
                data1 = data1.merge(data2, how='inner', indicator=True,
                                    left_on=key1, right_on=key2, copy=False)
                print data1
            else:
                data1 = data1.merge(data2, how=params['option'], indicator=True,
                                    left_on=key1, right_on=key2, copy=False)

        else:
            # create a temporary copy of the two dataframe
            # with the keys in lower caption
            data1_tmp = data1[key1].applymap(lambda col: str(col).lower())
            data1_tmp['data1_tmp'] = data1_tmp.index
            data2_tmp = data2[key2].applymap(lambda col: str(col).lower())
            data2_tmp['data2_tmp'] = data2_tmp.index

            if not last:
                data1_tmp = pd.merge(data1_tmp, data2_tmp, how='inner',
                                     left_on=key1, right_on=key2, copy=False)
            else:
                data1_tmp = pd.merge(data1_tmp, data2_tmp, how=params['option'],
                                     left_on=key1, right_on=key2, copy=False)

            data1_tmp = data1_tmp.drop(key1+key2, axis=1)
            data1 = pd.merge(data1, data1_tmp, left_index=True,
                             right_on='data1_tmp', copy=False, how='inner')

            data1 = data1.merge(data2, left_on='data2_tmp', indicator=True,
                                right_index=True, copy=False, how=params['option'])
            data1.drop(['data1_tmp', 'data2_tmp'], axis=1, inplace=True)

        if last:
            if params['option'] != 'inner':
                if len(data1) > 0 and len(result) > 0:
                    idx1 = data1.index
                    idxR = result.index
                    toRM = [i for i in idxR if i in idx1]
                    data1 = data1.drop(toRM)

        if len(result) > 0:
            data1 = pd.concat([result, data1])

        if last:
            if not keep:
                key2 += ['_merge']
                # remove all key columns of the second DataFrame
                data1 = data1.drop(key2, axis=1)
            else:
                data1 = data1.drop(['_merge'], axis=1)

        return data1


    @task(returns=list)
    def join_partial_sort(self, data, key):
        """Perform a partial sort to optimize the join."""
        data = data.dropna(axis=0, how='any', subset=key)
        N = len(data)
        cols = list(data.columns)
        if N > 0:
            data = data[key]
            data.sort_values(key, inplace=True)
            data.reset_index(drop=True, inplace=True)
            min_idx = data.loc[0, :].values.tolist()
            max_idx = data.loc[N-1, :].values.tolist()
            idx = [min_idx, max_idx, cols]
            return idx
        return [[], [], cols]


    @task(returns=list)
    def overlap(self, sortIdx1, sortIdx2, numFrag):
        """Check if fragments A and B may have some elements to be joined."""
        overlapping = \
            [[[False, False] for i in range(numFrag)] for i in range(numFrag)]

        for i in range(numFrag):
            x1, x2, cols1 = sortIdx1[i]
            if len(x1) != 0:  # only if data1 was empty
                for j in range(numFrag):
                    y1, y2, cols2 = sortIdx2[j]
                    if len(y1) != 0:  # only if data2 was empty

                        tmp = pd.DataFrame([x1, x2, y1, y2], index=[0, 1, 2, 3])
                        tmp = tmp.sort_values(by=0)
                        idx = tmp.index
                        # import numpy as np
                        # keys = np.array([str(x1[0]), str(x2[0]),
                        #                  str(y1[0]), str(y2[0])])
                        # index = np.array([0, 1, 2, 3])
                        # inds = keys.argsort()
                        # idx = index[inds]

                        # print tmp
                        # print idx
                        if any([idx[0] == 0 and idx[1] == 2,
                                idx[0] == 2 and idx[1] == 0]):
                                self.overlapping[i][j][0] = True

        for i in range(numFrag):
            for j, item in reversed(list(enumerate(overlapping[i]))):
                if item[0]:
                    overlapping[i][j][1] = True
                    break

        return overlapping


    @task(returns=list)
    def fix_columns(self, data1, cols2, params):
        """It is necessary change same columns names in empty partitions."""
        keep = params.get('keep_keys', False)
        suffixes = params.get('suffixes', ['_l', '_r'])
        key1 = params['key1']
        key2 = params['key2']
        LSuf = suffixes[0]
        RSuf = suffixes[1]

        if params['option'] == 'right':
            params['option'] = 'left'
            key1, key2 = key2, key1
            LSuf, RSuf = RSuf, LSuf

        cols1 = data1.columns
        cols2 = cols2[2]
        data2 = pd.DataFrame(columns=cols2)

        # Adding the suffixes before join.
        # This is necessary to preserve the keys
        # of the second table even though with equal name.
        convert1, key1 = self.rename_cols(cols1, cols2, key1, LSuf,  keep, True)
        data1 = data1.rename(columns=convert1)
        convert2, key2 = self.rename_cols(cols2, cols1, key2, RSuf, keep, False)
        data2 = data2.rename(columns=convert2)

        data1 = data1.merge(data2, how=params['option'],
                            left_on=key1, right_on=key2)

        if not keep:
            data1 = data1.drop(key2, axis=1)
        return data1
