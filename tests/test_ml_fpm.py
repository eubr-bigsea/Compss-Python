#!/usr/bin/python
# -*- coding: utf-8 -*-

from ddf_library.ddf import DDF
import pandas as pd


def ml_fpm_fpgrowth():
    from ddf_library.functions.ml.fpm import AssociationRules, FPGrowth

    data = pd.DataFrame([['1,2,5'],
                         ['1,2,3,5'],
                         ['1,2']], columns=['col_0'])

    data_set = DDF() \
        .parallelize(data, 2) \
        .map(lambda row: row['col_0'].split(','), 'col_0')

    # data_set = DDF()\
    #     .load_text('/transactions.csv', num_of_parts=4, header=False,
    #                sep='\n')\
    #     .map(lambda row: row['col_0'].split(' ')[:-1], 'col_0')

    fp = FPGrowth(min_support=0.5)  # 0.0284
    item_set = fp.fit_transform(data_set, column='col_0')

    """
             items  support
    0          [1]        3
    1          [2]        3
    2       [1, 2]        3
    3          [5]        2
    4       [1, 5]        2
    5      [ 2, 5]        2
    6   [1,  2, 5]        2
    """

    rules = AssociationRules(confidence=0.6).fit_transform(item_set)

    print('RESULT item set:')
    item_set.show()
    print("RESULT rules:")
    rules.show()


if __name__ == '__main__':
    print("_____Testing Frequent Pattern Mining_____")
    ml_fpm_fpgrowth()
