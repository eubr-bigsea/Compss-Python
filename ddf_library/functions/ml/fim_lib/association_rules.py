#!/usr/bin/env python3
# -*- coding: utf-8 -*-


__author__ = "Lucas Miguel S Ponce"
__email__ = "lucasmsp@gmail.com"

from ddf_library.ddf import DDF
from ddf_library.utils import generate_info, merge_info, read_stage_file, \
    save_stage_file, create_stage_files
from ddf_library.bases.ddf_model import ModelDDF
from ddf_library.functions.etl.hash_partitioner import hash_partition

from pycompss.api.parameter import FILE_IN, FILE_OUT
from pycompss.api.api import compss_wait_on, compss_delete_object
from pycompss.api.task import task

from itertools import chain, combinations
import pandas as pd


class AssociationRules(ModelDDF): # TODO
    # noinspection PyUnresolvedReferences
    """
    Association rule learning is a rule-based machine learning method for
    discovering interesting relations between variables in large databases.
    It is intended to identify strong rules discovered in databases.

    :Example:

    >>> rules = AssociationRules(confidence=0.10).fit_transform(item_set)
    """

    def __init__(self, confidence=0.5, max_rules=-1):

        """
        Setup all AssociationsRules's parameters.

        :param confidence: Minimum confidence (default is 0.5);
        :param max_rules: Maximum number of output rules, -1 to all (default).
        """
        super(AssociationRules, self).__init__()

        if not (0.0 <= confidence <= 1.0):
            raise Exception('Minimal confidence must be in '
                            'range [0, 1] but got {}'.format(confidence))
        self.confidence = confidence
        self.max_rules = max_rules
        self.col_item = None
        self.col_freq = None

    def set_min_confidence(self, confidence):
        self.confidence = confidence

    def set_max_rules(self, count):
        self.max_rules = count

    def fit_transform(self, data, col_item='items', col_freq='support'):
        """
        Fit the model.

        :param data: DDF;
        :param col_item: Column with the frequent item set (default, *'items'*);
        :param col_freq: Column with its support (default, *'support'*);
        :return: DDF with 'Pre-Rule', 'Post-Rule' and 'confidence' columns.
        """

        col_aux = 'auxiliary'
        self.col_item = col_item
        self.col_freq = col_freq
        # noinspection PyTypeChecker

        df, nfrag, tmp = self._ddf_initial_setup(data)

        info1 = [[] for _ in range(nfrag)]
        aux1, aux2 = info1[:], info1[:]
        info2, info = info1[:], info1[:]

        # TODO: save result in file
        for f in range(nfrag):
            aux1[f], info1[f], aux2[f], info2[f] = \
                _ar_flat_map_consequent(df[f], col_item, col_freq, f)

        info1, info2 = merge_info(info1), merge_info(info2)
        info1, info2 = compss_wait_on(info1), compss_wait_on(info2)

        # first, perform a hash partition to shuffle both data
        hash_params1 = {'columns': [col_aux], 'nfrag': nfrag, 'info': [info1],
                        'stage_id': -1}  # TODO
        hash_params2 = {'columns': [col_aux], 'nfrag': nfrag, 'info': [info2],
                        'stage_id': -1}
        output1 = hash_partition(aux1, hash_params1)
        output2 = hash_partition(aux2, hash_params2)
        out1, out2 = output1['data'], output2['data']

        result = create_stage_files(nfrag)
        for f in range(nfrag):
            info[f] = _ar_calculate_conf(out1[f], out2[f], result[f],
                                         self.confidence, f)

        # TODO: sort by confidence, sample, rebase ?
        if self.max_rules > 0:
            pass

        compss_delete_object(out1)
        compss_delete_object(out2)

        uuid_key = self._ddf_add_task(task_name='task_associative_rules',
                                      status='COMPLETED', opt=self.OPT_OTHER,
                                      result=result,
                                      function=self.fit_transform,
                                      parent=[tmp.last_uuid],
                                      info=info)

        return DDF(task_list=tmp.task_list, last_uuid=uuid_key)


@task(returns=4, data_input=FILE_IN)
def _ar_flat_map_consequent(data_input, col_item, col_freq, f):
    """Perform a partial rules generation."""
    df = read_stage_file(data_input, [col_freq, col_item])

    flat_map = []

    list_support = df[col_freq].values
    list_items = df[col_item].values

    df['auxiliary'] = ['_'.join(sorted(xi)) for xi in list_items]
    df = df[['auxiliary', col_freq]]
    df.rename({'support': 'den'}, axis='columns', inplace=True)

    for item, support in zip(list_items, list_support):
        if len(item) > 0:

            subsets = [list(x) for x in _ar_subsets(item)]
            for element in subsets:
                remain = list(set(item).difference(element))

                if len(remain) == 1:
                    element = sorted(element)
                    aux = "_".join(element)
                    row_i = [element, remain, support, aux]
                    flat_map.append(row_i)

    # num = Support(X union Y)  => in rules
    # den = Support(X)
    # confidence (X => Y) = Support (X union Y) / Support(X)
    cols = ['Pre-Rule', 'Post-Rule', 'num', 'auxiliary']
    rules = pd.DataFrame(flat_map, columns=cols)

    info1 = generate_info(rules, f)
    info2 = generate_info(df, f)

    return rules, info1, df, info2


@task(returns=1, data_input1=FILE_IN, data_input2=FILE_IN, data_output=FILE_OUT)
def _ar_calculate_conf(data_input1, data_input2, data_output,
                       confidence, f):
    data1 = read_stage_file(data_input1)
    data2 = read_stage_file(data_input2)

    data1 = data1.merge(data2, on='auxiliary', how='inner', copy=False)
    data1['confidence'] = data1['num'] / data1['den']

    data1.query('confidence >= {}'.format(confidence), inplace=True)
    data1 = data1[['Pre-Rule', 'Post-Rule', 'confidence']]

    info = generate_info(data1, f)
    save_stage_file(data1, data_output)
    return info


def _ar_subsets(arr):
    """Return non empty subsets of arr."""
    return chain(*[combinations(arr, i + 1) for i, a in enumerate(arr)])
