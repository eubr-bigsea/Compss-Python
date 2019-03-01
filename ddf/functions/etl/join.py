#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__ = "Lucas Miguel S Ponce"
__email__ = "lucasmsp@gmail.com"

from pycompss.api.task import task
from pycompss.functions.reduce import merge_reduce
import pandas as pd
import numpy as np


class JoinOperation(object):
    """Join Operation.

    Joins with another DataFrame, using the given join expression.
    """

    def preprocessing(self, params):
        key1 = params.get('key1', [])
        key2 = params.get('key2', [])
        option = params.get('option', 'inner')

        if any([len(key1) == 0,
                len(key2) == 0,
                len(key1) != len(key2),
                option not in ['inner', 'left', 'right']
                ]):
            raise \
                Exception('You must inform the keys of first '
                          'and second dataframe. You also must '
                          'inform the join type (inner,left or right join).')
        return key1, key2, option

    def transform(self, data1, data2, params, nfrag):
        """JoinOperation.

        :param data1: A list with nfrag pandas's dataframe;
        :param data2: Other list with nfrag pandas's dataframe;
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
            - 'sorted-data1':
            - 'sorted-data2':
        :param nfrag: The number of fragments;
        :return: Returns a list with nfrag pandas's dataframe.

        :Note: sort with case

        :note: Need schema as input
        """
        result = [[] for _ in range(nfrag)]
        info = [[] for _ in range(nfrag)]
        key1, key2, option = self.preprocessing(params)

        sorted_idx1, sorted_idx2 = _generate_idxs(data1, data2,
                                                  key1, key2, nfrag)

        if option == "right":
            overlapping = overlap(sorted_idx2, sorted_idx1, nfrag)

            for f1 in range(nfrag):
                filled = False
                for f2 in range(nfrag):
                    over, last = overlapping[f1][f2]
                    if over:
                        result[f1], info[f1] = \
                            _join(result[f1], data2[f1], data1[f2],
                                  params, last)
                        filled = True
                if not filled:
                    result[f1], info[f1] = fix_columns(data2[f1],
                                                       sorted_idx1[0], params)
        else:
            overlapping = overlap(sorted_idx1, sorted_idx2, nfrag)

            for f1 in range(nfrag):
                filled = False
                for f2 in range(nfrag):
                    over, last = overlapping[f1][f2]
                    if over:
                        result[f1], info[f1] = _join(result[f1], data1[f1],
                                                     data2[f2], params, last)
                        filled = True
                if not filled:
                    result[f1], info[f1] = fix_columns(data1[f1],
                                                       sorted_idx2[0], params)

        output = {'key_data': ['data'], 'key_info': ['info'],
                  'data': result, 'info': info}
        return output


def _generate_idxs(data1, data2, key1, key2, nfrag):
    sorted_idx1 = [[] for _ in range(nfrag)]
    sorted_idx2 = [[] for _ in range(nfrag)]

    for i in range(nfrag):
        sorted_idx1[i] = _join_partial_sort(data1[i], key1)
        sorted_idx2[i] = _join_partial_sort(data2[i], key2)

    sorted_idx1 = merge_reduce(_merge_idx, sorted_idx1)
    sorted_idx2 = merge_reduce(_merge_idx, sorted_idx2)

    from pycompss.api.api import compss_wait_on
    sorted_idx1 = compss_wait_on(sorted_idx1)
    sorted_idx2 = compss_wait_on(sorted_idx2)

    return sorted_idx1, sorted_idx2


def overlap(sorted_idx1, sorted_idx2, nfrag):
    """Check if fragments A and B may have some elements to be joined."""
    overlapping = \
        [[[False, False] for _ in range(nfrag)] for _ in range(nfrag)]
    # from pandas.api.types import is_categorical_dtype
    for i in range(nfrag):
        x1, x2, cols1 = sorted_idx1[i]
        if len(x1) != 0:  # only if data1 was empty
            for j in range(nfrag):
                y1, y2, cols2 = sorted_idx2[j]
                if len(y1) != 0:  # only if data2 was empty

                    tmp = pd.DataFrame([x1, x2, y1, y2], index=[0, 1, 2, 3])
                    tmp = tmp.infer_objects()

                    cols = [0]
                    # ndim = 0
                    # for c in tmp.columns:
                    #     if not is_categorical_dtype(tmp[c].dtype):
                    #         cols.append(c)
                    #     elif ndim == 0:
                    #         cols.append(c)
                    #         ndim += 1

                    tmp.sort_values(by=cols, inplace=True)
                    idx = tmp.index

                    if any([idx[0] == 0 and idx[1] == 2,
                            idx[0] == 2 and idx[1] == 0,
                            all(tmp.iloc[0, cols] == tmp.iloc[2, cols])]):
                            overlapping[i][j][0] = True

    for i in range(nfrag):
        for j, item in reversed(list(enumerate(overlapping[i]))):
            if item[0]:
                overlapping[i][j][1] = True
                break

    return overlapping


@task(returns=1)
def _join_partial_sort(data, key):
    """Perform a partial sort to optimize the join."""
    data = data.dropna(axis=0, how='any', subset=key)
    n = len(data)
    cols = list(data.columns)
    if n > 0:
        data = data[key]
        data.sort_values(key, inplace=True)
        data.reset_index(drop=True, inplace=True)
        min_idx = data.loc[0, :].values.tolist()
        max_idx = data.loc[n-1, :].values.tolist()
        idx = [min_idx, max_idx, cols]
        return [idx]
    return [[[], [], cols]]


@task(returns=1, priority=True)
def _merge_idx(idx1, idx2):
    return np.concatenate((idx1, idx2), axis=0)


def rename_cols(cols1, cols2, key, suf, keep, op):
    """Rename columns based in the columns of other dataset."""
    convert = {}
    for col in cols1:
        if col in cols2:
            if not keep and op and col in key:
                pass
            else:
                n_col = "{}{}".format(col, suf)
                convert[col] = n_col
                key = [n_col if x == col else x for x in key]

    return convert, key


def check_dtypes(data1, data2, key1, key2):

    data1[key1] = data1[key1].infer_objects()
    data2[key2] = data2[key2].infer_objects()

    from pandas.api.types import is_numeric_dtype
    for c1, c2 in zip(key1, key2):
        type1 = data1[c1].dtype
        type2 = data2[c2].dtype

        if type1 != type2:
            if is_numeric_dtype(data1[c1]) and is_numeric_dtype(data2[c2]):
                pd.to_numeric(data1[c1], downcast='float')
                pd.to_numeric(data2[c2], downcast='float')
            else:
                data1[c1] = data1[c1].astype(str)
                data2[c2] = data2[c2].astype(str)

    return data1, data2


@task(returns=2)
def _join(result, data1, data2, params, last):
    """Peform a join and a concatenation with the previosly join."""
    case_sensitive = params.get('case', True)
    keep = params.get('keep_keys', False)
    suffixes = params.get('suffixes', ['_l', '_r'])
    key1 = params['key1']
    key2 = params['key2']
    l_suf = suffixes[0]
    r_suf = suffixes[1]
    opt = params['option']

    if params['option'] == 'right':
        params['option'] = 'left'
        key1, key2 = key2, key1
        l_suf, r_suf = r_suf, l_suf

    cols1 = data1.columns
    cols2 = data2.columns

    # Adding the suffixes before join.
    # This is necessary to preserve the keys
    # of the second table even though with equal name.
    convert1, key1 = rename_cols(cols1, cols2, key1,  l_suf, keep, True)
    data1 = data1.rename(columns=convert1)
    convert2, key2 = rename_cols(cols2, cols1, key2, r_suf, keep, False)
    data2 = data2.rename(columns=convert2)

    # Removing rows where NaN is in keys
    data1 = data1.dropna(axis=0, how='any', subset=key1)
    data2 = data2.dropna(axis=0, how='any', subset=key2)

    data1, data2 = check_dtypes(data1, data2, key1, key2)

    if case_sensitive:
        if not last:
            data1 = data1.merge(data2, how='inner', indicator=True,
                                left_on=key1, right_on=key2, copy=False)
        else:
            opt = params['option']
            data1 = data1.merge(data2, how=opt, indicator=True,
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
            data1_tmp = pd.merge(data1_tmp, data2_tmp, how=opt,
                                 left_on=key1, right_on=key2, copy=False)

        data1_tmp = data1_tmp.drop(key1+key2, axis=1)
        data1 = pd.merge(data1, data1_tmp, left_index=True,
                         right_on='data1_tmp', copy=False, how='inner')

        data1 = data1.merge(data2, left_on='data2_tmp', indicator=True,
                            right_index=True, copy=False, how=opt)
        data1.drop(['data1_tmp', 'data2_tmp'], axis=1, inplace=True)

    if last:
        if opt != 'inner':
            if len(data1) > 0 and len(result) > 0:
                idx1 = data1.index
                idx_r = result.index
                to_rm = [i for i in idx_r if i in idx1]
                data1 = data1.drop(to_rm)

    if len(result) > 0:
        data1 = pd.concat([result, data1])

    if last:
        if not keep:
            key2 += ['_merge']
            # remove all key columns of the second DataFrame
            data1 = data1.drop(key2, axis=1)
        else:
            data1 = data1.drop(['_merge'], axis=1)

    print "[INFO - JOIN] - columns:", list(data1.columns)
    print "[INFO - JOIN] - length:", len(data1)

    info = [data1.columns.tolist(), data1.dtypes.values, [len(data1)]]
    return data1, info


@task(returns=2)
def fix_columns(data1, cols2, params):
    """It is necessary change same columns names in empty partitions."""
    keep = params.get('keep_keys', False)
    suffixes = params.get('suffixes', ['_l', '_r'])
    key1 = params['key1']
    key2 = params['key2']
    l_suf = suffixes[0]
    r_suf = suffixes[1]

    if params['option'] == 'right':
        params['option'] = 'left'
        key1, key2 = key2, key1
        l_suf, r_suf = r_suf, l_suf

    cols1 = data1.columns
    cols2 = cols2[2]
    data2 = pd.DataFrame(columns=cols2)

    # Adding the suffixes before join:  This is necessary to
    # preserve the keys of the second table even though with equal name.
    convert1, key1 = rename_cols(cols1, cols2, key1, l_suf,  keep, True)
    convert2, key2 = rename_cols(cols2, cols1, key2, r_suf, keep, False)
    data1 = data1.rename(columns=convert1)
    data2 = data2.rename(columns=convert2)

    data1, data2 = check_dtypes(data1, data2, key1, key2)

    data1 = data1.merge(data2, how=params['option'],
                        left_on=key1, right_on=key2)

    if not keep:
        data1 = data1.drop(key2, axis=1)

    info = [data1.columns.tolist(), data1.dtypes.values, [len(data1)]]
    return data1, info
