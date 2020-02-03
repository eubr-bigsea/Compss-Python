#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Lucas Miguel S Ponce"
__email__ = "lucasmsp@gmail.com"

import ddf_library.bases.config as config
from ddf_library.utils import read_stage_file

from pycompss.api.task import task
from pycompss.api.parameter import FILE_IN

from itertools import combinations
from collections import defaultdict, namedtuple


@task(returns=config.x, data_input=FILE_IN)
def step4_pfg(data_input, col, g_list, nfrag):
    """
    Parallel FP-Growth
    """
    g_list = g_list[0]
    result = [[] for _ in range(nfrag)]

    df = read_stage_file(data_input, col)
    for transaction in df[col].values:

        # group_list has already been pruned, but item_set hasn't
        item_set = [item for item in transaction if item in g_list]

        # for each transaction, sort item_set by count in descending order
        item_set = sorted(item_set,
                          key=lambda item: g_list[item].count,
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

                result[group_id].append(item_set[:(i + 1)])

    return result


def merge_n_reduce(f, data, n):
    """
    Apply f cumulatively to the items of data,
    from left to right in binary tree structure, so as to
    reduce the data to a single value.

    :param f: function to apply to reduce data
    :param data: List of items to be reduced
    :param n: step size
    :return: result of reduce the data to a single value
    """

    from collections import deque
    q = deque(range(len(data)))
    new_data = data[:]
    len_q = len(q)
    while len_q:
        x = q.popleft()
        len_q = len(q)
        if len_q:
            min_d = min([len_q, n - 1])
            xs = [q.popleft() for _ in range(min_d)]
            xs = [new_data[i] for i in xs] + [0] * (n - min_d - 1)

            new_data[x] = f(new_data[x], *xs)
            q.append(x)

        else:
            return new_data[x]


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
        for item in header_table:
            nodes = header_table[item]
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
    for i in range(1, len(x) + 1):
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

    for node in child_list:
        children = child_list[node]
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
