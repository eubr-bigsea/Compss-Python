#!/usr/bin/env python3
# -*- coding: utf-8 -*-


__author__ = "Lucas Miguel S Ponce"
__email__ = "lucasmsp@gmail.com"

import random
import numpy as np
import pandas as pd
from itertools import chain, combinations
from collections import defaultdict, namedtuple, Counter
from pycompss.api.task import task
from pycompss.functions.reduce import merge_reduce
from pycompss.api.api import compss_wait_on

from pycompss.api.local import local


from ddf_library.ddf import DDF, DDFSketch, generate_info

__all__ = ['AssociationRules', 'FPGrowth']

import sys
sys.path.append('../../')


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
        super(AssociationRules, self).__init__()
        df, nfrag, tmp = self._ddf_inital_setup(data)

        col_item = self.settings['col_item']
        col_freq = self.settings['col_freq']
        min_conf = self.settings['confidence']

        info = [[] for _ in range(nfrag)]
        result = [[] for _ in range(nfrag)]
        for f in range(nfrag):
            result[f], info[f] = _ar_get_rules(df, col_item, col_freq,
                                               f, min_conf)

        # if max_rules > -1:
        #     conf = ['confidence']
        #     rules_sorted = sort_byOddEven(df_rules, conf, nfrag)
        #     count = [_count_transations(rules_sorted[f]) for f in range(nfrag)]
        #     count_total = mergeReduce(_mergecount, count)
        #     for f in range(nfrag):
        #         df_rules[f] = _filter_rules(rules_sorted[f],
        #                                     count_total, max_rules, f)

        uuid_key = self._ddf_add_task(task_name='task_associative_rules',
                                      status='COMPLETED', lazy=False,
                                      function={0: result},
                                      parent=[tmp.last_uuid],
                                      n_output=1, n_input=1, info=info)

        self._set_n_input(uuid_key, 0)
        return DDF(task_list=tmp.task_list, last_uuid=uuid_key)


@task(returns=2)
def _ar_get_rules(df, col_item, col_freq, i, min_confidence):
    """Perform a partial rules generation."""
    list_rules = []
    for index, row in df[i].iterrows():
        item = row[col_item]
        support = row[col_freq]
        if len(item) > 0:
            subsets = [list(x) for x in _ar_subsets(item)]

            for element in subsets:
                remain = list(set(item).difference(element))

                if len(remain) > 0:
                    num = float(support)
                    den = _ar_get_support(element, df, col_item, col_freq)
                    confidence = num/den

                    if confidence > min_confidence:
                        r = [element, remain, confidence]
                        list_rules.append(r)

    cols = ['Pre-Rule', 'Post-Rule', 'confidence']
    rules = pd.DataFrame(list_rules, columns=cols)

    info = [cols, rules.dtypes.values, [len(rules)]]
    return rules, info



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


class FPGrowth(DDFSketch):
    """
    FPGrowth implements the FP-growth algorithm described in the paper
    LI et al., Mining requent patterns without candidate generation, where
    “FP” stands for frequent pattern. Given a dataset of transactions, the
    first step of FP-growth is to calculate item frequencies and identify
    frequent items.


    LI, Haoyuan et al. Pfp: parallel fp-growth for query recommendation.
    In: Proceedings of the 2008 ACM conference on Recommender systems.
    ACM, 2008. p. 107-114.

    :Example:

    >>> fp = FPGrowth(column='col_0', min_support=0.10).run(ddf1)
    >>> itemset = fp.get_frequent_itemsets()
    >>> rules = fp.generate_association_rules(confidence=0.1)
    """

    def __init__(self, column, min_support=0.5):
        """
        Setup all FPGrowth's parameters.

        :param column: Transactions feature name;
        :param min_support: minimum support value.
        """
        super(FPGrowth, self).__init__()
        self.settings = dict()
        self.settings['column'] = column
        self.settings['min_support'] = min_support

        self.model = {}
        self.name = 'FPGrowth'
        self.result = []

    def run(self, data):
        """
        Fit the model.

        :param data: DDF
        :return: a trained model
        """
        col = self.settings.get('column', [])
        min_support = self.settings.get('min_support', 0.5)

        df, nfrag, tmp = self._ddf_inital_setup(data)

        info = [step2_mapper(df_p, col, min_support, nfrag) for df_p in df]
        g_list = merge_reduce(step2_reduce, info)

        df_group = [[] for _ in range(nfrag)]
        df_group_aux = [{} for _ in range(nfrag)]
        for f in range(nfrag):
            df_group[f], df_group_aux[f] = step4_pfg(df[f], col, g_list, f)

        for f in range(nfrag):
            for f2 in range(nfrag):
                if f2 != f:
                    df_group[f], df_group_aux[f2] = \
                        step4_merge(df_group[f], df_group_aux[f2], f)

        for f in range(nfrag):
            df_group[f] = step5_mapper(df_group[f], g_list)

        df_group = merge_reduce(step5_reducer, df_group)
        result, info = step6(df_group, nfrag)

        self.model = {'data': result, 'info': info,
                      'tasklist': tmp.task_list, 'last_uuid': tmp.last_uuid}

        return self

    def get_frequent_itemsets(self):
        """
        Get the frequent item set generated by FP-Growth.

        :return: DDF
        """
        if len(self.result) == 0:

            if len(self.model) == 0:
                raise Exception("Model is not fitted.")

            uuid_key = self._ddf_add_task(task_name='task_fpgrowth',
                                          status='COMPLETED', lazy=False,
                                          function={0: self.model['data']},
                                          parent=[self.model['last_uuid']],
                                          n_output=1, n_input=1,
                                          info=self.model['info'])

            tmp = DDF(task_list=self.model['tasklist'], last_uuid=uuid_key)
            tmp._set_n_input(uuid_key, 0)
            self.result = [tmp]
            return tmp
        else:
            return self.result[0]

    def generate_association_rules(self, confidence=0.5, max_rules=-1):
        """
        Generate a DDF with the association rules.

        :param confidence: Minimum confidence (default is 0.5);
        :param max_rules: Maximum number of output rules, -1 to all (default);
        :return: DDF with 'Pre-Rule', 'Post-Rule' and 'confidence' columns.
        """

        df = self.get_frequent_itemsets()
        ar = AssociationRules(confidence=confidence, max_rules=max_rules)
        rules = ar.run(df)

        return rules


@task(returns=1)
def step2_mapper(data, col, min_support, nfrag):
    """
    Parallel Counting
    """

    if isinstance(col, list):
        col = col[0]

    n = len(data)
    item_set = list(chain.from_iterable(data[col].values))
    item_set = Counter(item_set)

    return [item_set, n, 0, nfrag, min_support]


@task(returns=1)
def step2_reduce(info1, info2):
    """
     Grouping Items
    """
    item_set1, n1, i1, nfrag, min_support = info1
    item_set2, n2, i2, _, _ = info2

    n = n1+n2
    i = i1+i2 + 1
    item_set = item_set1 + item_set2

    if i == (nfrag - 1):

        min_support = int(min_support * n)
        group_list = {}
        group_count = namedtuple('group_count', ['group', 'count'])

        for item, count in item_set.items():
            group_id = random.randint(0, nfrag - 1)

            if count >= min_support:
                group_list[item] = group_count(group_id, count)

        item_set = group_list

        print group_list

    return [item_set, n, i, nfrag, min_support]


@task(returns=2)
def step4_pfg(df, col, g_list, f):
    """
    Parallel FP-Growth
    """
    g_list = g_list[0]
    r1 = {f: []}
    r_others = {}

    for transaction in df[col].values:

        # group_list has already been pruned, but item_set hasn't

        item_set = [item for item in transaction if item in g_list]

        # for each transaction, sort item_set by count in descending order
        item_set = sorted(item_set, key=lambda item: g_list[item].count,
                          reverse=True)

        # a list of the groups for each item
        items = [g_list[item].group for item in item_set]

        emitted_groups = set()

        # iterate backwards through the ordered list of items in the transaction
        # for each distinct group, emit the transaction to that group-specific
        # reducer.

        for i, group_id in reversed(list(enumerate(items))):

            # we don't care about length 1 itemsets
            if i == 0:
                continue

            if group_id not in emitted_groups:
                emitted_groups.add(group_id)

                if group_id == f:
                    r1[group_id].append(item_set[:(i + 1)])
                else:
                    if group_id not in r_others:
                        r_others[group_id] = []
                    r_others[group_id].append(item_set[:(i + 1)])

    return r1, r_others


@task(returns=2)
def step4_merge(r1, r_others, id_group):

    keys = r_others.keys()
    if id_group in keys:
        r1[id_group].extend(r_others[id_group])
        r_others.pop(id_group, None)

    return r1, r_others


@task(returns=1)
def step5_mapper(transactions, g_list):
    min_support = g_list[4]

    keys = transactions.keys()
    patterns = {}

    g_list = g_list[0]
    for key in g_list:
        items = frozenset({key})
        patterns[items] = g_list[key].count

    if len(keys) > 0:
        key = keys[0]
        transactions = transactions[key]
        fp_tree, header_table = build_fp_tree(transactions)

        for pattern in fp_growth(fp_tree, header_table, None, min_support):
            items, support = pattern.get()
            items = frozenset(items)
            patterns[items] = support

    return patterns


@task(returns=1)
def step5_reducer(patterns1, patterns2):

    for key in patterns2.keys():
        if key in patterns1:
            patterns1[key] = max([patterns1[key], patterns2[key]])
        else:
            patterns1[key] = patterns2[key]
        del patterns2[key]

    return patterns1


@local
def step6(patterns, nfrag):

    patterns = pd.DataFrame([[list(k), s]
                             for k, s in patterns.items()],
                            columns=['items', 'support'])

    patterns = patterns.sort_values(by=['support'], ascending=[False])
    types = patterns.dtypes.values
    cols = patterns.columns.tolist()
    patterns = np.array_split(patterns, nfrag)

    info = [[cols, types, [len(d)]] for d in patterns]
    return patterns, info


class Node(object):

    ItemSupport = namedtuple('ItemSupport', ['item', 'support'])

    def __init__(self, item, support, parent, children):
        self.item = item
        self.support = support
        self.parent = parent
        self.children = children

    def get_item_support(self):
        return self.ItemSupport(self.item, self.support)

    def get_single_prefix_path(self):
        path = []
        node = self
        while node.children:
            if len(node.children) > 1:
                return None
            path.append(node.children[0].get_item_support())
            node = node.children[0]
        return path

    def clone(self, item=None, support=None, parent=None, children=None):
        return Node(self.item if item is None else item,
                    self.support if support is None else support,
                    self.parent if parent is None else parent,
                    self.children if children is None else children)

    def depth(self):
        if not self.children:
            return 1
        max_child_depth = 0
        for child in self.children:
            depth = child.depth()
            if depth > max_child_depth:
                max_child_depth = depth
        return 1 + max_child_depth

    def destroy(self):
        for child in self.children:
            child.destroy()
            child.parent = None
        self.children = []


class Pattern(object):

    def __init__(self, items=None):
        if items:
            self.items = set([x.item for x in items])
            if len(self.items) == len(items):
                self.support = min([x.support for x in items])
            else:
                deduped = defaultdict(int)
                for x in items:
                    deduped[x.item] += x.support
                self.support = min(deduped.values())
        else:
            self.items = []
            self.support = 0

    def __or__(self, other):
        if other is None:
            return self

        pattern = Pattern()
        pattern.items = self.items | other.items
        pattern.support = min(self.support, other.support)
        return pattern

    def __len__(self):
        return len(self.items)

    def get(self):
        return [list(self.items), self.support]


def build_fp_tree(transactions):
    fp_tree = Node(None, None, None, [])
    header_table = defaultdict(list)
    for transaction in transactions:
        insert_transaction(fp_tree, header_table, transaction)

    return fp_tree, header_table


def insert_transaction(fp_tree, header_table, transaction):
    current_node = fp_tree
    for item in transaction:
        found = False
        for child in current_node.children:
            if child.item == item:
                current_node = child
                current_node.support += 1
                found = True
                break
        if not found:
            new_node = Node(item, 1, current_node, [])
            current_node.children.append(new_node)
            header_table[item].append(new_node)
            current_node = new_node


def fp_growth(tree, header_table, previous_pattern, min_support):
    single_path = tree.get_single_prefix_path()

    if single_path:
        for combination in combinations_(single_path):
            pattern = Pattern(combination) | previous_pattern
            if pattern.support >= min_support and len(pattern) > 1:
                yield pattern
    else:
        for item, nodes in header_table.iteritems():
            pattern = Pattern(nodes) | previous_pattern

            if pattern.support >= min_support:
                if len(pattern) > 1:
                    yield pattern

                conditional_tree, conditional_header_table = \
                    get_conditional_tree(nodes, tree)
                if conditional_tree:
                    for pattern in fp_growth(conditional_tree,
                                             conditional_header_table,
                                             pattern, min_support):
                        yield pattern


def combinations_(x):
    for i in xrange(1, len(x) + 1):
        for c in combinations(x, i):
            yield c


def get_conditional_tree(nodes, tree):

    child_list = defaultdict(set)
    header_table = defaultdict(set)
    shadowed = set()

    for node in nodes:

        leaf = node
        first_pass = True

        while node.parent is not None:
            parent = node.parent
            if hasattr(parent, '_shadow'):
                if parent.parent is not None:
                    parent._shadow.support += leaf.support
            else:
                if parent.parent is None:
                    parent._shadow = parent.clone(children=[])
                else:
                    parent._shadow = parent.clone(support=leaf.support,
                                                  children=[])
                shadowed.add(parent)

            if first_pass:
                first_pass = False
            else:
                child_list[parent].add(node._shadow)
                header_table[node.item].add(node._shadow)

            node = parent

    for node, children in child_list.iteritems():
        node._shadow.children = list(children)

    def set_parents(shadow):
        if shadow.children:
            for child in shadow.children:
                child.parent = shadow
                set_parents(child)

    shadow_tree = tree._shadow

    for node in shadowed:
        del node._shadow

    if not shadow_tree.children:
        return None, None

    set_parents(shadow_tree)

    return shadow_tree, header_table