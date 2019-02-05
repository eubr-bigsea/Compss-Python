#!/usr/bin/python
# -*- coding: utf-8 -*-


__author__ = "Lucas Miguel S Ponce"
__email__ = "lucasmsp@gmail.com"


import numpy as np
import pandas as pd
from itertools import chain, combinations
from collections import defaultdict
from pycompss.api.task import task
from pycompss.api.parameter import INOUT
from pycompss.functions.reduce import merge_reduce
from pycompss.api.api import compss_wait_on
from ddf import ddf

__all__ = ['AssociationRules', 'Apriori']


import uuid
import sys
sys.path.append('../../')


class AssociationRules(object):

    def __init__(self, col_item='items', col_freq='support',
                 confidence=0.5, max_rules=-1):

        self.settings = dict()
        self.settings['col_freq'] = col_freq
        self.settings['col_item'] = col_item
        self.settings['confidence'] = confidence
        self.settings['max_rules'] = max_rules

        self.model = []
        self.name = 'AssociationRules'

    def set_min_confidence(self, confidence):
        self.settings['confidence'] = confidence

    def run(self, data):
        """

        :param data: DDF
        :return:
        """

        nfrag = len(data.partitions[0])

        col_item = self.settings['col_item']
        col_freq = self.settings['col_freq']
        min_conf = self.settings['confidence']
        df_rules = [[] for _ in range(nfrag)]
        for i in range(nfrag):
            df_rules[i] = _ar_get_rules(data.partitions[0],
                                        col_item, col_freq, i, min_conf)

        # if max_rules > -1:
        #     conf = ['confidence']
        #     rules_sorted = sort_byOddEven(df_rules, conf, nfrag)
        #     count = [_count_transations(rules_sorted[f]) for f in range(nfrag)]
        #     count_total = mergeReduce(_mergecount, count)
        #     for f in range(nfrag):
        #         df_rules[f] = _filter_rules(rules_sorted[f],
        #                                     count_total, max_rules, f)

        data.partitions = {0: df_rules}
        uuid_key = str(uuid.uuid4())
        ddf.COMPSsContext.tasks_map[uuid_key] = {'name': 'task_associative_rules',
                                                 'status': 'COMPLETED',
                                                 'lazy': False,
                                                 'function': df_rules,
                                                 'parent': [data.last_uuid],
                                                 'output': 1, 'input': 1}

        data.set_n_input(uuid_key, data.settings['input'])
        return ddf.DDF(data.partitions, data.task_list, uuid_key)


@task(returns=list)
def _ar_filter_rules(rules, count, max_rules, pos):
    """Select the first N rules."""
    total, partial = count
    if total > max_rules:
        gets = 0
        for i in range(pos):
            gets += partial[i]
        number = max_rules-gets
        if number > partial[pos]:
            number = partial[pos]
        if number < 0:
            number = 0
        return rules.head(number)
    else:
        return rules


@task(returns=list)
def _ar_get_rules(freq_items, col_item, col_freq, i, min_confidence):
    """Perform a partial rules generation."""
    list_rules = []
    for index, row in freq_items[i].iterrows():
        item = row[col_item]
        support = row[col_freq]
        if len(item) > 0:
            subsets = [list(x) for x in _ar_subsets(item)]

            for element in subsets:
                remain = list(set(item).difference(element))

                if len(remain) > 0:
                    num = float(support)
                    den = _ar_get_support(element, freq_items, col_item, col_freq)
                    confidence = num/den

                    if confidence > min_confidence:
                        r = [element, remain, confidence]
                        list_rules.append(r)

    cols = ['Pre-Rule', 'Post-Rule', 'confidence']
    rules = pd.DataFrame(list_rules, columns=cols)

    return rules


def _ar_subsets(arr):
    """Return non empty subsets of arr."""
    return chain(*[combinations(arr, i + 1) for i, a in enumerate(arr)])


def _ar_get_support(element, freq_items, col_item, col_freq):
    """Retrive the support of an item."""
    for df in freq_items:
        for t, s in zip(df[col_item].values, df[col_freq].values):
            if element == t:
                return s
    return float("inf")


@task(returns=list)
def _ar_mergeRules(rules1, rules2):
    """Merge partial rules."""
    return pd.concat([rules1, rules2])


@task(returns=list)
def _ar_count_transations(rules):
    """Count the number of rules."""
    return [len(rules), [len(rules)]]


@task(returns=list)
def _ar_mergecount(c1, c2):
    """Merge the partial count."""
    return [c1[0]+c2[0], c1[1]+c2[1]]


class Apriori(object):

    def __init__(self, column='', min_support=0.5):

        self.settings = dict()
        self.settings['column'] = column
        self.settings['min_support'] = min_support

        self.model = []
        self.name = 'Apriori'

    def run(self, data):
        """

        :param data: DDF
        :return:
        """
        col = self.settings.get('column', [])
        minSupport = self.settings.get('min_support', 0.5)
        large_set = []

        df = data.partitions[0]
        nfrag = len(df)

        # pruning candidates
        currentCSet_reduced = \
            getFirstItemsWithMinSupport(df, col, minSupport, nfrag)
        currentCSet_merged = \
            merge_reduce(_merge_sets_in_local, currentCSet_reduced)
        currentCSet_merged = compss_wait_on(currentCSet_merged)

        k = 2
        while len(currentCSet_merged[0]) > 0:
            large_set.append(currentCSet_merged[0])
            currentLSet = joinSet(currentCSet_reduced, k, nfrag)
            currentCSet_reduced = \
                getItemsWithMinSupport(currentLSet, df, col,
                                       minSupport, nfrag)
            currentCSet_merged = \
                merge_reduce(_merge_sets_in_local, currentCSet_reduced)
            currentCSet_merged = compss_wait_on(currentCSet_merged)
            k += 1

        def removeFrozen(a):
            return list(a[0]), a[1]

        for i in range(len(large_set)):
            large_set[i] = np.array(large_set[i].items())
            large_set[i] = np.apply_along_axis(removeFrozen, 1, large_set[i])

        large_set = np.vstack((large_set))
        large_set = np.array_split(large_set, nfrag)

        result = []
        for subset in large_set:
            result.append(pd.DataFrame(subset, columns=['items', 'support']))

        self.model = [result, data.task_list, data.last_uuid]

        return self

    def get_frequent_itemsets(self):

        if len(self.model) == 0:
            raise Exception("Model is not fitted.")

        result = {0: self.model[0]}

        uuid_key = str(uuid.uuid4())
        df = ddf.DDF(result, self.model[1], uuid_key)
        ddf.COMPSsContext.tasks_map[uuid_key] = {'name': 'task_apriori',
                                                 'status': 'COMPLETED',
                                                 'lazy': False,
                                                 'function': result,
                                                 'parent': [self.model[2]],
                                                 'output': 1, 'input': 1}

        df.set_n_input(uuid_key, df.settings['input'])
        return df

    def generate_association_rules(self, confidence=0.5, max_rules=-1):

        df = self.get_frequent_itemsets()
        ar = AssociationRules(confidence=confidence, max_rules=max_rules)
        rules = ar.run(df)

        return rules


def getItemsWithMinSupport(candidates, data, col, minSupport, nfrag):
    """Return all candidates that meets a minimum support level."""
    sets_local = \
        [_count_items(candidates[i], data[i], col)
         for i in range(nfrag)]
    sets_global = merge_reduce(merge_ItemsWithMinSupport, sets_local)
    C_reduced = [[] for _ in range(nfrag)]
    for i in range(nfrag):
        C_reduced[i] = FilterItemsWithMinSupport(data[i], col,
                                                 sets_global, minSupport)

    return C_reduced


@task(returns=list)
def _count_items(items_set, data, col):
    """Count the frequency of an item."""
    if len(col) == 0:
        col = data.columns[0]

    local_set = defaultdict(int)

    for transaction in data[col].values:
        for item in items_set:
            if item.issubset(transaction):
                local_set[item] += 1

    return [local_set, len(data)]


def getFirstItemsWithMinSupport(data, col, min_support, nfrag):
    """Return all candidates that meets a minimum support level."""
    sets_local = [count_FirstItems(data[i], col) for i in range(nfrag)]
    sets_global = merge_reduce(merge_ItemsWithMinSupport, sets_local)
    C_reduced = [[] for _ in range(nfrag)]
    for i in range(nfrag):
        C_reduced[i] = FilterItemsWithMinSupport(data[i], col,
                                                 sets_global, min_support)

    return C_reduced


@task(returns=list)
def count_FirstItems(data, col):
    """Count the frequency of an item."""
    if len(col) == 0:
        col = data.columns[0]

    itemSet = set()
    for record in data[col].values:
        for item in record:
            itemSet.add(frozenset([item]))

    localSet = defaultdict(int)

    for transaction in data[col].values:
        for item in itemSet:
            if item.issubset(transaction):
                localSet[item] += 1

    return [localSet, len(data)]


@task(returns=list)
def merge_ItemsWithMinSupport(data1, data2):
    """Merge supports."""
    localSet1, n1 = data1
    localSet2, n2 = data2

    for freq in localSet2:
        if freq in localSet1:
            localSet1[freq] += localSet2[freq]
        else:
            localSet1[freq] = localSet2[freq]

    return [localSet1, n1+n2]


@task(returns=list)
def FilterItemsWithMinSupport(data, col, freq_set, min_support):
    """FilterItemsWithMinSupport.

    Calculates the support for items in the itemSet and returns a subset
    of the itemSet each of whose elements satisfies the minimum support
    """
    if len(col) == 0:
        col = data.columns[0]

    global_set, size = freq_set
    freq_tmp = defaultdict(int)

    for transaction in data[col].values:
        for item in global_set:
            if item.issubset(transaction):
                support = float(global_set[item])/size
                if support >= min_support:
                    freq_tmp[item] = support

    return [freq_tmp, size]


def joinSet(itemSets, length, nfrag):
    """Join a set with itself and returns the n-element itemsets."""
    joined = [[] for _ in range(nfrag)]
    for i in range(nfrag):
        tmp = [joiner(itemSets[i], itemSets[j], length)
               for j in range(nfrag) if i != j]
        joined[i] = merge_reduce(mergeSets, tmp)

    return joined


@task(returns=list)
def mergeSets(items_set1, items_set2):
    """Merge item sets."""
    return items_set1 | items_set2


@task(returns=list)
def _merge_sets_in_local(items_set1, items_set2):
    """Merge the two itemsets."""
    for k, v in items_set2[0].items():
        items_set1[0][k] = v

    return [items_set1[0], items_set1[1]]


@task(returns=list)
def joiner(itemSets_local1, itemSets_local2, length):
    """Merge only distintics items."""
    sets1 = set(itemSets_local1[0].keys())
    sets2 = set(itemSets_local2[0].keys())
    itemSets_global = sets1 | sets2
    joined = set([i.union(j) for i in itemSets_global
                  for j in itemSets_global if len(i.union(j)) == length])

    return joined


