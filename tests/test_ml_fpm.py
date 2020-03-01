#!/usr/bin/python
# -*- coding: utf-8 -*-

from ddf_library.context import COMPSsContext
import pandas as pd


def ml_fpm_fpgrowth():
    from ddf_library.functions.ml.fpm import AssociationRules, FPGrowth

    data = pd.DataFrame([['1,2,5'],
                         ['1,2,3,5'],
                         ['1,2']], columns=['col_0'])

    from ddf_library.columns import col, udf
    from ddf_library.types import DataType

    def f1(x):
        return x.split(',')
    cc = COMPSsContext()
    f1_udf = udf(f1, DataType.ARRAY, col('col_0'))
    data_set = cc \
        .parallelize(data, 2) \
        .map(f1_udf, 'col_0')

    def f2(x):
        return x.split(' ')[:-1]

    f2_udf = udf(f2, DataType.ARRAY, col('col_0'))

    # data_set = DDF()\
    #     .read.csv('hdfs://localhost:9000/transactions.csv',
    #               num_of_parts='*', header=False, sep='\n')\
    #     .map(f2_udf, 'col_0')

    fp = FPGrowth(min_support=0.5)  # 0.0284
    item_set = fp.fit_transform(data_set, input_col='col_0')

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

    # rules = AssociationRules(confidence=0.6).fit_transform(item_set)

    print('RESULT item set:')
    item_set.show()
    print("RESULT rules:")
    # rules.show()
    cc.stop()


if __name__ == '__main__':
    print("_____Testing Frequent Pattern Mining_____")
    ml_fpm_fpgrowth()
