#!/usr/bin/python
# -*- coding: utf-8 -*-
"""Apriori.

Run the apriori algorithm.
"""
__author__ = "Lucas Miguel S Ponce"
__email__ = "lucasmsp@gmail.com"

from itertools import chain, combinations
from collections import defaultdict
from pycompss.api.task import task
from pycompss.functions.reduce import mergeReduce


class Apriori(object):

    def runApriori(self, data, settings, nfrag):
        """runApriori.

        :param data:          A list of pandas dataframe.;
        :param settings:      A dictionary with the informations:
          - 'column':         Field with the transactions
                              (empty, first attribute);
          - 'minSupport':     Minimum support value (default, 0.5);
        :return               A list of pandas dataframe.
        """
        col = settings.get('column', [])
        minSupport = settings.get('minSupport', 0.5)
        large_set = []

        from pycompss.api.api import compss_wait_on

        # pruning candidates
        currentCSet_reduced = \
            getFirstItemsWithMinSupport(data, col, minSupport, nfrag)
        currentCSet_merged = \
            mergeReduce(_merge_sets_in_local, currentCSet_reduced)
        currentCSet_merged = compss_wait_on(currentCSet_merged)

        k = 2
        while len(currentCSet_merged[0]) > 0:
            large_set.append(currentCSet_merged[0])
            currentLSet = joinSet(currentCSet_reduced, k, nfrag)
            currentCSet_reduced = \
                getItemsWithMinSupport(currentLSet, data, col,
                                       minSupport, nfrag)
            currentCSet_merged = \
                mergeReduce(_merge_sets_in_local, currentCSet_reduced)
            currentCSet_merged = compss_wait_on(currentCSet_merged)
            k += 1

        import pandas as pd
        import numpy as np

        def removeFrozen(a):
            return list(a[0]), a[1]

        for i in range(len(large_set)):
            large_set[i] = np.array(large_set[i].items())
            large_set[i] = np.apply_along_axis(removeFrozen, 1, large_set[i])

        large_set = np.vstack((large_set))
        large_set = np.array_split(large_set, nfrag)

        df = []
        for subset in large_set:
            df.append(pd.DataFrame(subset, columns=['items', 'support']))

        return df

    def generateRules(self, subset, settings):
        """generateRules.

        Generates the list of rules in the form: predecessor, successor
        and its confidence for each item passed by parameter.

        :param settings:   A dictionary with the informations:
          - 'confidence':  The minimum confidence (default, 0.5);
        :return            A list of pandas dataframe.
        """
        min_confidence = settings.get('confidence', 0.5)
        rules = [get_rules(i, subset, min_confidence)
                 for i in range(len(subset))]
        return rules


def getItemsWithMinSupport(candidates, data, col, minSupport, nfrag):
    """Return all candidates that meets a minimum support level."""
    sets_local = \
        [_count_items(candidates[i], data[i], col)
         for i in range(nfrag)]
    sets_global = mergeReduce(merge_ItemsWithMinSupport, sets_local)
    C_reduced = [[] for i in range(nfrag)]
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
    sets_global = mergeReduce(merge_ItemsWithMinSupport, sets_local)
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
        joined[i] = mergeReduce(mergeSets, tmp)

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

# -------


@task(returns=list)
def get_rules(i, L, min_confidence):
    """Generate the partial rulers."""
    toRetRules = []
    for index, row in L[i].iterrows():
        item = row['items']
        support = row['support']
        if len(item) > 0:
            _subsets = [list(x) for x in subsets(item)]

            for element in _subsets:
                remain = list(set(item).difference(element))

                if len(remain) > 0:
                    num = float(support)
                    den = _get_support(element, L)
                    confidence = num/den

                    if confidence > min_confidence:
                        r = [element, remain, confidence]
                        toRetRules.append(r)

    import pandas as pd
    names = ['Pre-Rule', 'Post-Rule', 'confidence']
    rules = pd.DataFrame(toRetRules, columns=names)

    return rules


def subsets(arr):
    """Return non empty subsets of arr."""
    return chain(*[combinations(arr, i + 1) for i, a in enumerate(arr)])


def _get_support(element, sets):
    """Get support of a item. Return 'inf' if dont exist."""
    for df in sets:
        for t, s in zip(df['items'].values, df['support'].values):
            if element == t:
                return s
    return float("inf")
