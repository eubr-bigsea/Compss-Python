#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Lucas Miguel S Ponce"
__email__ = "lucasmsp@gmail.com"

from ddf_library.utils import generate_info, create_auxiliary_column

from pycompss.api.task import task
from pycompss.functions.reduce import merge_reduce

import pandas as pd
import numpy as np


class JoinOperation(object):
    """Joins with another DataFrame, using the given join expression."""

    @staticmethod
    def preprocessing(params):
        key1 = params.get('key1', [])
        key2 = params.get('key2', [])
        option = params.get('option', 'inner')
        params['option'] = option

        if any([len(key1) == 0,
                len(key2) == 0,
                len(key1) != len(key2),
                option not in ['inner', 'left', 'right']
                ]):
            raise \
                Exception('You must inform the keys of first '
                          'and second DataFrame. You also must '
                          'inform the join type (inner, left or right join).')
        return key1, key2, params

    def transform(self, data1, data2, settings):
        """
        :param data1: A list of pandas's DataFrame;
        :param data2: Other list of pandas's DataFrame;
        :param settings: A dictionary that contains:
            - 'option': 'inner' to InnerJoin, 'left' to left join and
                        'right' to right join.
            - 'key1': A list of keys of the first DataFrame;
            - 'key2': A list of keys of the second DataFrame;
            - 'case': True to case-sensitive (default, True);
            - 'keep_keys': True to keep the keys of the second data set,
                           (default, False).
            - 'suffixes': Suffixes for attributes, a list with 2 values
                          (default, [_l,_r]);
        :return: Returns a list of pandas's DataFrame.
        """

        key1, key2, settings = self.preprocessing(settings)
        nfrag1, nfrag2 = len(data1), len(data2)
        info1, info2 = settings['info'][0], settings['info'][1]
        nfrag = max([nfrag1, nfrag2])

        # first, perform a hash partition to shuffle both data
        from .repartition import hash_partition
        hash_params1 = {'columns': key1, 'nfrag': nfrag, 'info': [info1]}
        hash_params2 = {'columns': key2, 'nfrag': nfrag, 'info': [info2]}
        output1 = hash_partition(data1, hash_params1)
        output2 = hash_partition(data2, hash_params2)
        out1, out2 = output1['data'], output2['data']

        # second, pair-wise join
        result, info = [[] for _ in range(nfrag)], [[] for _ in range(nfrag)]
        for f in range(nfrag):
            result[f], info[f] = _join(out1[f], out2[f], settings, f)

        output = {'key_data': ['data'], 'key_info': ['info'],
                  'data': result, 'info': info}
        return output


@task(returns=2)
def _join(data1, data2, params, frag):
    """Perform a join and a concatenation with the previously join."""
    case_sensitive = params.get('case', True)
    keep = params.get('keep_keys', False)
    suffixes = params.get('suffixes', ['_l', '_r'])
    key1, key2 = params['key1'], params['key2']
    l_suf, r_suf = suffixes[0], suffixes[1]
    opt = params['option']

    # clean data: rename some columns and remove NaN values
    cols1, cols2 = list(data1.columns), list(data2.columns)
    data1, data2, key1, key2 = clean_df(data1, data2,
                                        key1, key2,
                                        l_suf, r_suf,
                                        cols1, cols2,
                                        keep)

    if case_sensitive:
        data1 = data1.merge(data2,
                            left_on=key1, right_on=key2,
                            how=opt, copy=False)

    else:
        # create a temporary copy of the two DataFrame
        # with the keys in lower caption
        aux_col1 = create_auxiliary_column(cols1)
        aux_col2 = create_auxiliary_column(cols2)
        data1_tmp = data1[key1]
        data2_tmp = data2[key2]
        for col in data1_tmp:
            data1_tmp[col] = data1_tmp[col].astype(str).str.lower()
        for col in data2_tmp:
            data2_tmp[col] = data2_tmp[col].astype(str).str.lower()

        data1_tmp[aux_col1] = data1_tmp.index
        data2_tmp[aux_col2] = data2_tmp.index

        data1_tmp = pd.merge(data1_tmp, data2_tmp,
                             left_on=key1, right_on=key2,
                             copy=False, how=opt)

        data1_tmp = data1_tmp.drop(key1+key2, axis=1)
        data1 = pd.merge(data1, data1_tmp,
                         left_index=True, right_on=aux_col1,
                         copy=False, how='inner')

        data1 = data1.merge(data2,
                            left_on=aux_col2, right_index=True,
                            copy=False, how=opt)
        data1.drop([aux_col1, aux_col2], axis=1, inplace=True)

    if not keep:
        # remove all key columns of the second DataFrame
        if opt == 'right':
            convert = {k2: k1 for k1, k2 in zip(key1, key2)}
            data1 = data1.drop(key1, axis=1)
            data1 = data1.rename(columns=convert)
        else:
            data1 = data1.drop(key2, axis=1)

    info = generate_info(data1, frag)
    return data1, info


def clean_df(data1, data2, key1, key2, l_suf, r_suf, cols1, cols2, keep):

    # Adding the suffixes before join. This is necessary to preserve the keys
    # of the second table even though with equal name.
    data1, key1 = rename_cols(data1, cols1, cols2, key1, l_suf, keep, True)
    data2, key2 = rename_cols(data2, cols2, cols1, key2, r_suf, keep, False)

    # Removing rows where NaN is in keys
    data1.dropna(axis=0, how='any', subset=key1, inplace=True)
    data2.dropna(axis=0, how='any', subset=key2, inplace=True)
    return data1, data2, key1, key2


def rename_cols(data, cols1, cols2, key, suf, keep, op):
    """Rename columns based in the columns of other data set."""
    convert = {}
    for col in cols1:
        if col in cols2:
            if not keep and op and col in key:
                pass
            else:
                n_col = "{}{}".format(col, suf)
                convert[col] = n_col
                key = [n_col if x == col else x for x in key]

    data = data.rename(columns=convert)
    return data, key
