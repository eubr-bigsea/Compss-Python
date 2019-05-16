#!/usr/bin/env python3
# -*- coding: utf-8 -*-


__author__ = "Lucas Miguel S Ponce"
__email__ = "lucasmsp@gmail.com"

from ddf_library.ddf import DDF, generate_info
from ddf_library.ddf_base import DDFSketch

from pycompss.api.task import task
from pycompss.functions.reduce import merge_reduce
from pycompss.api.api import compss_wait_on
# from pycompss.api.local import local

from itertools import chain, combinations
from collections import defaultdict, namedtuple, Counter
import random
import numpy as np
import pandas as pd


class AssociationRules(DDFSketch):
    """
    Association rule learning is a rule-based machine learning method for
    discovering interesting relations between variables in large databases.
    It is intended to identify strong rules discovered in databases using
    some measures of interestingness.

    :Example:

    >>> rules = AssociationRules(confidence=0.10).run(itemset)
    """

    def __init__(self, col_item='items', col_freq='support',
                 confidence=0.5, max_rules=-1):

        """
        Setup all AssociationsRules's parameters.

        :param col_item: Column with the frequent item set (default, *'items'*);
        :param col_freq: Column with its support (default, *'support'*);
        :param confidence: Minimum confidence (default is 0.5);
        :param max_rules: Maximum number of output rules, -1 to all (default).
        """
        super(AssociationRules, self).__init__()

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
        Fit the model.

        :param data: DDF
        :return: DDF with 'Pre-Rule', 'Post-Rule' and 'confidence' columns
        """

        df, nfrag, tmp = self._ddf_initial_setup(data)

        col_item = self.settings['col_item']
        col_freq = self.settings['col_freq']
        min_conf = self.settings['confidence']

        info = [[] for _ in range(nfrag)]
        result = [[] for _ in range(nfrag)]
        for f in range(nfrag):
            result[f], info[f] = _ar_get_rules(df, col_item, col_freq,
                                               f, min_conf)

        # if max_rules > -1:
        #     rules_sorted = sort_byOddEven(df_rules, conf, nfrag)
        #     count = [_count_transations(rules_sorted[f]) for f in range(nfrag)]
        #     count_total = mergeReduce(_mergecount, count)
        #     for f in range(nfrag):
        #         df_rules[f] = _filter_rules(rules_sorted[f],
        #                                     count_total, max_rules, f)

        uuid_key = self._ddf_add_task(task_name='task_associative_rules',
                                      status='COMPLETED', opt=self.OPT_OTHER,
                                      function={0: result},
                                      parent=[tmp.last_uuid],
                                      n_output=1, n_input=1, info=info)

        self._set_n_input(uuid_key, 0)
        return DDF(task_list=tmp.task_list, last_uuid=uuid_key)


@task(returns=2)
def _ar_get_rules(df, col_item, col_freq, f, min_confidence):
    """Perform a partial rules generation."""
    list_rules = []
    for index, row in df[f].iterrows():
        item = row[col_item]
        support = row[col_freq]
        if len(item) > 0:
            subsets = [list(x) for x in _ar_subsets(item)]

            for element in subsets:
                remain = list(set(item).difference(element))

                if len(remain) > 0:
                    den = _ar_get_support(element, df, col_item, col_freq)
                    confidence = support/den

                    if confidence > min_confidence:
                        r = [element, remain, confidence]
                        list_rules.append(r)

    cols = ['Pre-Rule', 'Post-Rule', 'confidence']
    rules = pd.DataFrame(list_rules, columns=cols)

    info = generate_info(rules, f)
    return rules, info


@task(returns=1)
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


@task(returns=1)
def _ar_mergeRules(rules1, rules2):
    """Merge partial rules."""
    return pd.concat([rules1, rules2])


@task(returns=1)
def _ar_count_transations(rules):
    """Count the number of rules."""
    return [len(rules), [len(rules)]]


@task(returns=1)
def _ar_mergecount(c1, c2):
    """Merge the partial count."""
    return [c1[0]+c2[0], c1[1]+c2[1]]

