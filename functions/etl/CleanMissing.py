#!/usr/bin/python
# -*- coding: utf-8 -*-
"""Clean Missing Operation: Clean missing fields from data set."""
__author__ = "Lucas Miguel S Ponce"
__email__ = "lucasmsp@gmail.com"

from pycompss.api.task import task
from pycompss.api.parameter import *
from pycompss.functions.reduce import mergeReduce
import pandas as pd
import math

class CleanMissingOperation(object):

    def __init__(self):
        pass

    def transform(self, data, params, nFrag):
        """CleanMissingOperation.

        :param data:          A list with numFrag pandas's dataframe;
        :param params:        A dictionary that contains:
            - attributes:     A list of attributes to evaluate;
            - cleaning_mode:  What to do with missing values;
              * "VALUE":         replace by parameter "value";
              * "REMOVE_ROW":    remove entire row (default);
              * "MEDIAN":        replace by median value;
              * "MODE":          replace by mode value;
              * "MEAN":          replace by mean value;
              * "REMOVE_COLUMN": remove entire column;
            - value:         Value to replace missing values (if mode is "VALUE");
        :param nFrag:      The number of fragments;
        :return:             Returns a list with numFrag pandas's dataframe.

        example:
            settings['attributes']    =  ["ID_POSONIBUS"]
            settings['cleaning_mode'] =  "VALUE"
            settings['value'] = -1
        """
        cleaning_mode = params.get('cleaning_mode', 'REMOVE_ROW')
        params['cleaning_mode'] = cleaning_mode

        if cleaning_mode in ['VALUE', 'REMOVE_ROW']:
            # we dont need to take in count others rows/fragments
            data = [self._clean_missing(data[f], params)
                    for f in range(nFrag)]
        else:
            params = [self._clean_missing_pre(data[f], params)
                      for f in range(nFrag)]
            params = mergeReduce(self.merge_clean_options, params)
            data = [self._clean_missing(data[f], params)
                    for f in range(nFrag)]

        return data


    @task(returns=dict)
    def _clean_missing_pre(self, data, params):
        """REMOVE_COLUMN, MEAN, MODE and MEDIAN needs pre-computation."""
        attributes = params['attributes']
        cleaning_mode = params['cleaning_mode']

        if cleaning_mode == "REMOVE_COLUMN":
            # list of columns of the current fragment that contains a null value
            null_fields = \
                data[attributes].columns[data[attributes].isnull().any()].tolist()
            params['columns_drop'] = null_fields

        elif cleaning_mode == "MEAN":
            # generate a partial mean of each subset column
            params['values'] = data[attributes].mean().values

        elif cleaning_mode in ["MODE", 'MEDIAN']:
            # generate a frequency list of each subset column
            dict_mode = {}
            for att in attributes:
                dict_mode[att] = data[att].value_counts()
            params['dict_mode'] = dict_mode

        return params


    @task(returns=dict)
    def merge_clean_options(self, params1, params2):
        """Merge pre-computations."""
        cleaning_mode = params1['cleaning_mode']

        if cleaning_mode == "REMOVE_COLUMN":
            params1['columns_drop'] = \
                list(set(params1['columns_drop'] + params2['columns_drop']))

        elif cleaning_mode in "MEAN":
            params1['values'] = \
                [(x + y)/2 for x, y in zip(params1['values'],
                                           params2['values'])]

        elif cleaning_mode in ["MODE", 'MEDIAN']:
            dict_mode1 = params1['dict_mode']
            dict_mode2 = params2['dict_mode']
            dict_mode = {}
            for att in dict_mode1:
                dict_mode[att] = \
                    pd.concat([dict_mode1[att], dict_mode2[att]], axis=1).\
                    fillna(0).sum(axis=1)
            params1['dict_mode'] = dict_mode

        return params1


    @task(returns=list)
    def _clean_missing(self, data, params):
        """Perform REMOVE_ROW, REMOVE_COLUMN, VALUE, MEAN, MODE and MEDIAN."""
        attributes = params['attributes']
        cleaning_mode = params['cleaning_mode']

        if cleaning_mode == "REMOVE_ROW":
            data.dropna(axis=0, how='any', subset=attributes, inplace=True)

        elif cleaning_mode == "VALUE":
            value = params['value']
            data[attributes] = data[attributes].fillna(value=value)

        elif cleaning_mode == "REMOVE_COLUMN":
            subset = params['columns_drop']
            data = data.drop(subset, axis=1)

        elif cleaning_mode == "MEAN":
            values = params['values']
            for v, a in zip(values, attributes):
                data[a] = data[a].fillna(value=v)

        elif cleaning_mode == "MODE":
            dict_mode = params['dict_mode']
            for att in dict_mode:
                # t = dict_mode[att].max()
                mode = dict_mode[att].idxmax()
                data[att] = data[att].fillna(value=mode)

        elif cleaning_mode == "MEDIAN":
            dict_mode = params['dict_mode']
            for att in dict_mode:
                total = dict_mode[att].sum()
                if total % 2 == 0:
                    m1 = total/2
                    m2 = total/2 + 1
                    count = 0
                    for i, p in enumerate(dict_mode[att]):
                        count += p
                        if count >= m1:
                            v1 = dict_mode[att].index[i]
                        if count >= m2:
                            v2 = dict_mode[att].index[i]
                            break
                    m = (float(v1)+float(v2))/2
                else:
                    m = math.floor(float(total)/2) + 1
                    count = 0
                    for i, p in enumerate(dict_mode[att]):
                        count += p
                        if count >= m:
                            v1 = dict_mode[att].index[i]
                            break
                    m = float(v1)

                data[att] = data[att].fillna(value=m)

        data.reset_index(drop=True, inplace=True)
        return data
