#!/usr/bin/env python3
# -*- coding: utf-8 -*-


__author__ = "Lucas Miguel S Ponce"
__email__ = "lucasmsp@gmail.com"

from ddf_library.ddf import DDF
from ddf_library.utils import generate_info, read_stage_file, \
    create_stage_files, save_stage_file
from ddf_library.bases.ddf_model import ModelDDF

from pycompss.api.parameter import FILE_IN, COLLECTION_IN
from pycompss.api.task import task
from pycompss.functions.reduce import merge_reduce
from pycompss.api.api import compss_wait_on

from itertools import chain
from collections import namedtuple, Counter
import random
import numpy as np
import pandas as pd
import importlib


class FPGrowth(ModelDDF):
    # noinspection PyUnresolvedReferences
    # noinspection SpellCheckingInspection
    """
    FPGrowth implements the FP-growth algorithm described in the paper
    LI et al., Mining frequent patterns without candidate generation, where
    “FP” stands for frequent pattern. Given a data set of transactions, the
    first step of FP-growth is to calculate item frequencies and identify
    frequent items.

    LI, Haoyuan et al. Pfp: parallel fp-growth for query recommendation.
    In: Proceedings of the 2008 ACM conference on Recommender systems.
    ACM, 2008. p. 107-114.

    :Example:

    >>> fp = FPGrowth(min_support=0.10)
    >>> item_set = fp.fit_transform(ddf1, column='col_0')
    """

    def __init__(self, min_support=0.5):
        """
        Setup all FPGrowth's parameters.

        :param min_support: minimum support value.
        """
        super(FPGrowth, self).__init__()
        self.min_support = min_support

    def fit_transform(self, data, input_col):
        """
        Fit the model and transform the data.

        :param data: DDF;
        :param input_col: Transactions feature name;
        :return: DDF
        """

        df, nfrag, new_data = self._ddf_initial_setup(data)
        self.input_col = input_col

        # stage 1 and 2: Parallel Counting and Grouping Items
        info = [step2_mapper(df_p, input_col,
                             self.min_support, nfrag) for df_p in df]
        g_list = step2_reduce(info)

        # stage 3 and 4: Parallel FP-Growth
        splits = [[[] for _ in range(nfrag)] for _ in range(nfrag)]
        df_group = [[] for _ in range(nfrag)]

        import ddf_library.bases.config
        ddf_library.bases.config.x = nfrag

        import ddf_library.functions.ml.fim_lib.fp_growth_aux
        importlib.reload(ddf_library.functions.ml.fim_lib.fp_growth_aux)

        for f in range(nfrag):
            splits[f] = ddf_library.functions.ml.fim_lib.\
                fp_growth_aux.step4_pfg(df[f], input_col, g_list, nfrag)

        for f in range(nfrag):
            tmp = [splits[f2][f] for f2 in range(nfrag)]
            df_group[f] = step4_merge_and_step5(tmp, g_list)

        df_group = step5_reducer(df_group)
        # split the result in nfrag to keep compatibility with others algorithms
        result, info = step6(df_group, nfrag)

        uuid_key = self._ddf_add_task(task_name=self.name,
                                      status=self.STATUS_MATERIALIZED,
                                      opt=self.OPT_OTHER,
                                      function=self.fit_transform,
                                      result=result,
                                      parent=[new_data.last_uuid],
                                      info=info)

        return DDF(task_list=new_data.task_list, last_uuid=uuid_key)


@task(returns=1, data_input=FILE_IN)
def step2_mapper(data_input, col, min_support, nfrag):
    """
    Parallel Counting
    """
    if isinstance(col, list):
        col = col[0]
    df = read_stage_file(data_input, cols=[col])
    n = len(df)
    item_set = list(chain.from_iterable(df[col].to_numpy().tolist()))
    item_set = Counter(item_set)

    return [item_set, n, nfrag, min_support]


@task(returns=1, partial_counts=COLLECTION_IN)
def step2_reduce(partial_counts):
    """
    Grouping Items
    """
    item_set, n, nfrag, min_support = partial_counts[0]
    for pc in partial_counts[1:]:
        item_set_tmp, n_tmp, _, _ = pc
        item_set += item_set_tmp
        n += n_tmp

    min_support = round(min_support * n)
    group_list = {}
    group_count = namedtuple('group_count', ['group', 'count'])

    for item, count in item_set.items():
        group_id = random.randint(0, nfrag - 1)

        if count >= min_support:
            group_list[item] = group_count(group_id, count)

    return [group_list, min_support]


@task(returns=1, results=COLLECTION_IN)
def step4_merge_and_step5(results, g_list_info):
    transactions = []
    for r in results:
        if r != 0:
            transactions.extend(r)
    del results

    from .fp_growth_aux import build_fp_tree, fp_growth
    g_list, min_support = g_list_info
    patterns = {}

    keys = list(g_list.keys())  # list of items

    for key in g_list:
        items = frozenset({key})
        patterns[items] = g_list[key].count

    if len(keys) > 0:
        fp_tree, header_table = build_fp_tree(transactions)

        for pattern in fp_growth(fp_tree, header_table, None, min_support):
            items, support = pattern.get()
            patterns[frozenset(items)] = support

    return patterns


@task(returns=1, patterns=COLLECTION_IN)
def step5_reducer(patterns):

    pattern = patterns[0]
    patterns = patterns[1:]
    for pattern_tmp in patterns:
        for key in list(pattern_tmp.keys()):
            if key in pattern:
                pattern[key] = max([pattern[key], pattern_tmp[key]])
            else:
                pattern[key] = pattern_tmp[key]
            del pattern_tmp[key]

    return pattern


def step6(patterns, nfrag):
    # currently, we assume that the final list of patterns can be fit in memory
    patterns = compss_wait_on(patterns)
    patterns = pd.DataFrame([[list(k), s] for k, s in patterns.items()],
                            columns=['items', 'support'])

    patterns = patterns.sort_values(by=['support'], ascending=[False])

    data_out = create_stage_files(nfrag)
    patterns = np.array_split(patterns, nfrag)
    info = [generate_info(patterns[f], f) for f in range(nfrag)]
    for (out, df) in zip(data_out, patterns):
        save_stage_file(out, df)
    return data_out, info
