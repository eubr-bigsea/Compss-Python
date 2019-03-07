#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__ = "Lucas Miguel S Ponce"
__email__ = "lucasmsp@gmail.com"

from pycompss.api.task import task
from pycompss.functions.reduce import merge_reduce
from pycompss.api.local import local
import pandas as pd
import numpy as np
import copy


class DropNaN(object):

    def __init__(self, subset=None, how='any', thresh=None, mode=0):
        """

        :param subset:  optional list of column names to consider.
        :param thresh: int, default None If specified, drop rows that have less
            than thresh non-null values. This overwrites the how parameter.
        :param how: ‘any’ or ‘all’. If ‘any’, drop a row if it contains any
         nulls. If ‘all’, drop a row only if all its values are null.
        :return: Returns a list with pandas's DataFrame.
        """

        self.settings = {'cleaning_mode': mode,
                         'attributes': subset,
                         'how': how, 'thresh': thresh}

    def preprocessing(self, data):

        # we need to take in count others rows/fragments
        params = [_clean_missing_pre(df, self.settings) for df in data]
        self.settings = merge_reduce(merge_clean_options, params)

        return self

    def drop_rows(self, data):

        subset = self.settings['attributes']
        thresh = self.settings['thresh']
        how = self.settings['how']
        data.dropna(how=how, subset=subset, thresh=thresh, inplace=True)

        data.reset_index(drop=True, inplace=True)
        info = [data.columns.tolist(), data.dtypes.values, [len(data)]]
        return data, info

    def drop_columns(self, data):
        settings = copy.deepcopy(self.settings)
        nfrag = len(data)
        result = [[] for _ in range(nfrag)]
        info = [[] for _ in range(nfrag)]

        for f in range(nfrag):
            result[f], info[f] = _clean_missing(data[f], settings)

        output = {'key_data': ['data'], 'key_info': ['info'],
                  'data': result, 'info': info}
        return output


class FillNa(object):

    def __init__(self, subset=None, mode='VALUE', value=None):

        self.settings = {'cleaning_mode': mode,
                         'attributes': subset,
                         'value': value}

    def preprocessing(self, data):

        if self.settings['cleaning_mode'] is not 'MEDIAN':
            # we need to generate mean value
            params = [_clean_missing_pre(df, self.settings) for df in data]
            self.settings = merge_reduce(merge_clean_options, params)
        else:
            """
            Based on : 
            FUJIWARA, Akihiro; INOUE, Michiko; MASUZAWA, Toshimitsu. Parallel 
            selection algorithms for CGM and BSP models with application to 
            sorting. IPSJ Journal, v. 41, p. 1500-1508, 2000.
            """
            # 1- On each processor, find the median of all elements on the
            # processor.
            stage1 = [_median_stage1(df, self.settings) for df in data]
            stage1 = merge_reduce(_median_stage1_merge, stage1)

            # 2- Select the median of the medians. Let M M be the median of the
            # medians.
            nfrag = len(data)
            info = [[] for _ in range(nfrag)]
            result = [[] for _ in range(nfrag)]
            for f in range(nfrag):
                info[f], result[f] = _median_stage2(data[f], stage1)
            info_stage2 = merge_reduce(_median_stage2_merge, info)

            # 3 Broadcast M M to all processors.
            # 4- Split the elements on each processor into two subsets, L and U.
            # The subset L contains elements that are smaller than MM, and
            # the subset U contains elements that are larger than MM .
            # 5 - Compute SUM L
            for f in range(nfrag):
                info[f] = _median_stage3(result[f], info_stage2)
            info = merge_reduce(_median_stage3_merge, info)

            self.settings['values'] = _median_define(info)

        return self

    def fill_by_value(self, data):
        value = self.settings['value']
        subset = self.settings['attributes']

        if isinstance(value, dict):
            data.fillna(value=value, inplace=True)
        else:
            if not subset:
                subset = data.columns.tolist()
            values = {key: value for key in subset}
            data.fillna(value=values, inplace=True)

        data.reset_index(drop=True, inplace=True)
        info = [data.columns.tolist(), data.dtypes.values, [len(data)]]
        return data, info

    def fill_by_statistic(self, data):

        settings = copy.deepcopy(self.settings)

        nfrag = len(data)
        result = [[] for _ in range(nfrag)]
        info = [[] for _ in range(nfrag)]

        for f in range(nfrag):
            result[f], info[f] = _clean_missing(data[f], settings)

        output = {'key_data': ['data'], 'key_info': ['info'],
                  'data': result, 'info': info}
        return output


@task(returns=1)
def _median_stage1(df, params):

    subset = params['attributes']
    dict_median = {}
    for att in subset:
        x = [x for x in df[att].values if ~np.isnan(x)]
        size = len(x)
        if size > 0:
            median = [np.median(x)]
        else:
            median = []
        dict_median[att] = [size, median]

    return dict_median


@task(returns=1)
def _median_stage1_merge(dict_median1, dict_median2):
    for att in dict_median2:
        if att not in dict_median1:
            dict_median1[att] = dict_median2[att]
        else:
            s1, m1 = dict_median1[att]
            s2, m2 = dict_median2[att]
            dict_median1[att] = [s1+s2, m1+m2]

    return dict_median1


@task(returns=2)
def _median_stage2(df, dict_median):

    u_l_list = {}
    info = {}
    for att in dict_median:
        size, medians = dict_median[att]
        median_of_medians = np.median(medians)
        x = [x for x in df[att].values if ~np.isnan(x)]
        low_list = [low for low in x if low <= median_of_medians]
        upper_list = [high for high in x if high >= median_of_medians]
        sum_l = len(low_list)
        sum_u = len(upper_list)
        u_l_list[att] = [low_list, upper_list]

        info[att] = [median_of_medians, size, sum_l, sum_u]

    return info, u_l_list


@task(returns=1)
def _median_stage2_merge(info1, info2):

    for att in info2:
        if att not in info1:
            info1[att] = info2[att]
        else:
            median_of_medians1, size1, sum_l1, sum_u1 = info1[att]
            _, _, sum_l2, sum_u2 = info2[att]
            info1[att] = [median_of_medians1, size1,
                          sum_l1 + sum_l2, sum_u1 + sum_u2]

    return info1


@task(returns=1)
def _median_stage3(u_l_list, info):

    for att in info:

        median_of_medians, size, sum_l, sum_u = info[att]

        cond = 0
        ith = float(size) / 2
        if size % 2 == 0:
            last = 2
            ith = ith + 0.5
        else:
            last = 1

        low, high = u_l_list[att]
        if ith < sum_l:
            low = sorted(low)[-last:]
            high = sorted(high)[:1]
            info[att] = [last, low, 0, high, cond]

        elif ith == sum_l:

            info[att] = [1, [median_of_medians], 1, [], cond]

        else:
            if ith >= sum_u:
                cond = 1
            high = sorted(high)[:last]
            low = sorted(low)[-1:]
            info[att] = [last, high, 2, low, cond]

    return info


@task(returns=1)
def _median_stage3_merge(info1, info2):

    for att in info2:
        if att not in info1:
            info1[att] = info2[att]
        else:
            last, num_p2, op, num_o2, cond = info2[att]
            _, num_p1, _, num_o1, _ = info1[att]

            if op == 2:
                nums1 = sorted(num_p1 + num_p2)[:last]
                nums2 = sorted(num_o1 + num_o2)
                if len(nums2) == 0:
                    nums2 = []
                else:
                    nums2 = [nums2[-1]]
            else:
                nums1 = sorted(num_p1 + num_p2)[-last:]
                nums2 = sorted(num_o1 + num_o2)
                if len(nums2) == 0:
                    nums2 = []
                else:
                    nums2 = [nums2[0]]

            info1[att] = [last, nums1, op, nums2, cond]

    return info1


@local
def _median_define(info):
    for att in info:

        last, nums1, op, nums2, cond = info[att]
        if last == 2:
            if cond:
                nums = nums2[0] + nums1[0]
            else:
                nums = sum(nums1)
        else:
            nums = sum(nums1)
        info[att] = float(nums) / last

    return info


@task(returns=1)
def _clean_missing_pre(data, params):
    """REMOVE_COLUMN, MEAN, MODE and MEDIAN needs pre-computation."""
    subset = params['attributes']
    cleaning_mode = params['cleaning_mode']
    thresh = params.get('thresh', None)
    how = params.get('how', None)

    if cleaning_mode == "REMOVE_COLUMN":
        # list of columns of the current fragment
        # that contains a null value

        if thresh:
            null_fields = data[subset].isnull().sum()
        elif how == 'any':
            null_fields = \
                data[subset].columns[data[subset].isnull().any()].tolist()
        else:
            null_fields = \
                data[subset].columns[data[subset].isnull().all()].tolist()

        params['columns_drop'] = null_fields

    elif cleaning_mode == "MEAN":
        # generate a partial mean of each subset column
        params['values'] = [len(data),
                            data[subset].sum(numeric_only=True,
                                             skipna=True).values]

    elif cleaning_mode in "MODE":
        # generate a frequency list of each subset column
        dict_mode = {}
        for att in subset:
            dict_mode[att] = data[att].value_counts()
        params['dict_mode'] = dict_mode
        print dict_mode
    return params


@task(returns=1)
def merge_clean_options(params1, params2):
    """Merge pre-computations."""
    cleaning_mode = params1['cleaning_mode']
    thresh = params1.get('thresh', None)
    how = params1.get('how', None)

    if cleaning_mode == "REMOVE_COLUMN":
        drops1 = params1['columns_drop']
        drops2 = params2['columns_drop']
        if thresh:
            params1['columns_drop'] = drops1 + drops2
        elif how == 'any':
            params1['columns_drop'] = list(set(drops1 + drops2))
        else:
            params1['columns_drop'] = list(set(drops1).intersection(drops2))

    elif cleaning_mode is "MEAN":
        size1, sums1 = params1['values']
        size2, sums2 = params2['values']
        params1['values'] = [size1+size2, sums1 + sums2]

    elif cleaning_mode is "MODE":
        dict_mode1 = params1['dict_mode']
        dict_mode2 = params2['dict_mode']
        dict_mode = {}
        for att in dict_mode1:
            dict_mode[att] = \
                pd.concat([dict_mode1[att], dict_mode2[att]], axis=0).\
                fillna(0).sum(level=0)
        params1['dict_mode'] = dict_mode

    return params1


@task(returns=2)
def _clean_missing(data, params):
    """Perform REMOVE_ROW, REMOVE_COLUMN, VALUE, MEAN, MODE and MEDIAN."""
    attributes = params['attributes']
    cleaning_mode = params['cleaning_mode']

    if cleaning_mode == "REMOVE_COLUMN":

        thresh = params['thresh']
        if thresh:
            subset = []
            cols = params['columns_drop']
            for c in cols.index:
                if cols.loc[c] > thresh:
                    subset.append(c)
        else:
            subset = params['columns_drop']
        data = data.drop(subset, axis=1)

    elif cleaning_mode == "MEAN":

        sizes, sums = params['values']
        values = np.divide(sums, sizes)

        for v, a in zip(values, attributes):
            data[a] = data[a].fillna(value=v)

    elif cleaning_mode == "MODE":
        dict_mode = params['dict_mode']
        for att in dict_mode:
            mode = dict_mode[att].idxmax()
            data[att] = data[att].fillna(value=mode)

    elif cleaning_mode == 'MEDIAN':
        print params
        medians = params['values']
        for att in medians:
            data[att] = data[att].fillna(value=medians[att])

    data.reset_index(drop=True, inplace=True)
    info = [data.columns.tolist(), data.dtypes.values, [len(data)]]
    return data, info
