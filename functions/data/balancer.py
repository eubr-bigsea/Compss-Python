#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__ = "Lucas Miguel S Ponce"
__email__ = "lucasmsp@gmail.com"

from pycompss.api.task import task
from pycompss.api.parameter import INOUT
from pycompss.functions.reduce import mergeReduce
import numpy as np
import pandas as pd


class WorkloadBalancerOperation(object):
    """Redistribute the data in equal parts if it's unbalanced. 
    It is considered an unbalanced dataframe if the coefficient of 
    variation (CV) between fragments is greater than 0.20."""
    
    def transform(self, data, forced, nfrag):
        """WorkloadBalancerOperation.
        
            :param data: A list with nfrag pandas's dataframe;
            :param forced: True to force redistribution of data, False to use
                heuristic based on the CV;
            :param nfrag: The number of fragments;
            :return: Returns a balanced list with nfrag pandas's dataframe.
        """

        balanced, len1, total = self.preprocessing(data, forced, nfrag)
        result = data[:]
        if not balanced:
            new_size = int(np.ceil(float(total)/nfrag))

            for f in range(nfrag - 1):
                to_transfer = new_size - len1[f]
    
                if to_transfer > 0:
                    # If lines are missing, get from the next fragments
                    for g in xrange(f + 1, nfrag):
                        # amount that the next fragment can yield
                        if len1[g] >= to_transfer:
                            offset = to_transfer
                        else:
                            offset = len1[g]
    
                        data[f] = balancing_f2_to_f1(result[f],
                                                     result[g], offset)
                        len1[g] -= offset
                        len1[f] += offset
                        to_transfer -= offset
    
                elif to_transfer < 0:
                    # if lines are in excess, move it to the next block
                    data[f + 1] = balancing_f1_to_f2(result[f],
                                                     result[f + 1],
                                                     -to_transfer)
                    len1[f + 1] -= to_transfer
                    len1[f] += to_transfer
    
            return data

    def preprocessing(self, data, forced, nfrag):
        """Check the distribution of the data."""
        len1 = [_counting_records(data[f]) for f in range(nfrag)]
        len1 = mergeReduce(_merging_count, len1)
        from pycompss.api.api import compss_wait_on
        len1 = compss_wait_on(len1)
        total = len1[0]
        len1 = len1[1]

        if forced:
            balanced = False
        else:
            cv = np.std(len1) / np.mean(len1)
            print "Coefficient of variation:{}".format(cv)
            if cv > 0.20:
                balanced = False
            else:
                balanced = True

        return balanced, len1, total


@task(returns=list)
def _counting_records(data):
    return [len(data), [len(data)]]


@task(returns=list)
def _merging_count(len1, len2):
    return [len1[0]+len2[0], len1[1]+len2[1]]


@task(df_f1=INOUT, returns=list)
def balancing_f1_to_f2(df_f1, df_f2, off1):
    # Get the tail offset lines from df_f1
    # and put at the head of df_f2
    tmp = df_f1.tail(off1)
    df_f1.drop(tmp.index, inplace=True)
    tmp.reset_index(drop=True, inplace=True)

    mynparray = df_f2.values
    mynparray = np.vstack((tmp, mynparray))
    df_f2 = pd.DataFrame(mynparray, columns=df_f2.columns)
    return df_f2


@task(df_f2=INOUT, returns=list)
def balancing_f2_to_f1(df_f1, df_f2, offset):
    # Get the head offset lines from df_f2
    # and put at the tail of df_f1
    tmp = df_f2.head(offset)
    df_f2.drop(tmp.index, inplace=True)
    tmp.reset_index(drop=True, inplace=True)

    mynparray = df_f1.values
    mynparray = np.vstack((mynparray, tmp))
    df_f1 = pd.DataFrame(mynparray, columns=df_f1.columns)
    return df_f1
