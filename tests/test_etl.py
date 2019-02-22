#!/usr/bin/python
# -*- coding: utf-8 -*-

from ddf.ddf import DDF
import pandas as pd
import numpy as np


def etl():

    url = ('https://archive.ics.uci.edu/ml/'
           'machine-learning-databases/abalone/abalone.data')
    cols = ['sex', 'length', 'diam', 'height', 'weight', 'rings']
    data = pd.read_csv(url, usecols=[0, 1, 2, 3, 4, 8], names=cols)[:20]

    f1 = lambda col: -42 if col['height'] > 0.090 else col['height']
    data1 = DDF().parallelize(data, 4)\
                 .transform(f1, 'height_nan').cache()

    data2 = data1.transform(lambda col: "."+col['sex'], 'sex')\
                 .cache()

    print """
    ----------------------------------------------------------------------------
        etl_test_1: Avaliar a capacidade de multiplos caches
        """

    df = data1.cache().show()
    print "etl_test_1a:", df[0:5]

    df = data2.cache().show()
    print "etl_test_1b:", df[0:5]

    df = data1.cache().show()
    print "etl_test_1c:", df[0:5]

    print """
    ----------------------------------------------------------------------------
        etl_test_2: Avaliar a capacidade de branching:
        data2 e data3 são filhos de data1
        nesse caso: nenhum dos transforms das listas podem ser otimizadas
    """

    data3 = data1.drop(['length', 'diam']).cache()
    print "DATA1", data1.task_list
    print "DATA3", data3.task_list
    df1 = data1.show()
    df2 = data2.show()
    df3 = data3.show()

    print "etl_test_2a:", df1[0:5]
    print "etl_test_2b:", df2[0:5]
    print "etl_test_2c:", df3[0:5]

    print """
    ----------------------------------------------------------------------------
          etl_test_3: Avaliar a capacidade de agrupar multiplas lazy tasks
          Nesse caso, drop, drop e replace vao ser agrupadas
    """
    data4 = data2.drop(['length']).drop(['diam'])\
        .replace({'rings': [[15], [42]]})

    df = data4.cache().show()
    print "etl_test_3:", df[0:5]

    print """
    ----------------------------------------------------------------------------
          etl_test_4: Avaliar a capacidade de produção de dois resultados
    """

    data5a, data5b = data4.split(0.5)

    df = data5b.cache().show()
    print "etl_test_4a:", df[0:5]

    df = data5a.cache().show()
    print "etl_test_4b:", df[0:5]

    print """
    ----------------------------------------------------------------------------
        etl_test_5: Avaliar capacidade de esperar uma segunda entrada
    """

    data6 = data5b.join(data5a, ['rings'], ['rings'])\
                  .select(['sex_l', 'height_l', 'weight_l', 'rings'])\
                  .filter('(rings > 8)')

    df = data6.cache().show()
    print "etl_test_5a len({}): {}".format(len(df), df[0:5])

    data7 = data6.sample(10).sort(['rings'], [True])
    data8 = data6.join(data7, ['rings'], ['rings'])

    print "data8", data8.task_list
    df = data8.cache().show()
    print "etl_test_5b len({}): {}".format(len(df), df[0:5])

    print """
    ----------------------------------------------------------------------------
        etl_test_6: Avaliar capacidade de gerar resultado sem salvar varivel
    """
    df = data1.distinct(['rings']).cache().show()
    df2 = data1.cache().show()
    print "etl_test_6a:", df
    print "etl_test_6b:", df2

    print """
    ----------------------------------------------------------------------------
        etl_test_7: Avaliar capacidade 'count' and 'take'
    """

    v = data1.select(['rings']).count()
    df = data1.select(['rings']).take(10).cache().show()

    print "etl_test_7a:", v
    print "etl_test_7b:", len(df)


def simple_etl():
    from pandas.util.testing import assert_frame_equal

    data = pd.DataFrame([[i, i+5, 0] for i in range(10)],
                        columns=['a', 'b', 'c'])

    data1 = pd.DataFrame([[i, i + 5] for i in range(5)],
                         columns=['a', 'b'])

    data2 = pd.DataFrame([[i, i + 5, 0] for i in xrange(5, 15)],
                         columns=['a', 'b', 'c'])

    print "\n|-------- Add Column --------|\n"
    ddf_1a = DDF().parallelize(data1, 5) # 2
    ddf_1b = DDF().parallelize(data2, 10)
    df1 = ddf_1a.add_column(ddf_1b).show()

    res_add = pd.DataFrame([[0,     5,     5, 10,  0],
                            [1,     6,     6, 11,  0],
                            [2,     7,     7, 12,  0],
                            [3,     8,     8, 13,  0],
                            [4,     9,     9, 14,  0],
                            [None, 10,  None, 15,  0],
                            [None, 11,  None, 16,  0],
                            [None, 12,  None, 17,  0],
                            [None, 13,  None, 18,  0],
                            [None, 14,  None, 19,  0]
                            ], columns=['a_l', 'a_r', 'b_l', 'b_r', 'c'])
    print df1.equals(res_add)
    #assert_frame_equal(df1, res_add, check_index_type=False) #, check_dtype=False)
    print df1

    return 0
    print "etl_test - add column - OK",

    print "\n|-------- Aggregation --------|\n"
    express = {'a': ['count'], 'b': ['first', 'last']}
    aliases = {'a': ["COUNT"], 'b': ['col_First', 'col_Last']}
    ddf_1 = DDF().parallelize(data, 4)\
        .aggregation(['c'], exprs=express, aliases=aliases)
    df1 = ddf_1.cache().show()

    res_agg = pd.DataFrame([[0, 10, 5, 14]],
                           columns=['c', 'COUNT', 'col_First', 'col_Last'])
    assert_frame_equal(df1, res_agg, check_index_type=False)
    print "etl_test - aggregation - OK",

    print "\n|-------- Difference --------|\n"
    ddf_1a = DDF().parallelize(data, 4)
    ddf_1b = DDF().parallelize(data2, 4)
    ddf_2 = ddf_1a.difference(ddf_1b)
    df1 = ddf_2.cache().show()
    res_diff = pd.DataFrame([[0, 5, 0], [1, 6, 0], [2, 7, 0],
                             [3, 8, 0], [4, 9, 0]], columns=['a', 'b', 'c'])
    assert_frame_equal(df1, res_diff, check_index_type=False)
    print "etl_test - difference - OK"

    print "\n|-------- Distinct --------|\n"
    ddf_1 = DDF().parallelize(data, 4).distinct(['c'])
    df1 = ddf_1.cache().show()

    res_dist = pd.DataFrame([[0, 5, 0]], columns=['a', 'b', 'c'])
    assert_frame_equal(df1, res_dist, check_index_type=False)
    print "etl_test - distinct - OK"

    print "\n|-------- Drop --------|\n"
    ddf_1 = DDF().parallelize(data, 4).drop(['a'])
    df1 = ddf_1.cache().show()
    res_drop = pd.DataFrame([[5, 0], [6, 0], [7, 0], [8, 0], [9, 0],
                             [10, 0], [11, 0], [12, 0],
                             [13, 0], [14, 0]], columns=['b', 'c'])
    assert_frame_equal(df1, res_drop, check_index_type=False)
    print "etl_test - drop - OK"

    print "\n|-------- Intersect --------|\n"
    ddf_1a = DDF().parallelize(data, 4)
    ddf_1b = DDF().parallelize(data2, 4)
    ddf_2 = ddf_1a.intersect(ddf_1b)
    df1 = ddf_2.cache().show()
    res_int = pd.DataFrame([[5, 10, 0], [6, 11, 0], [7, 12, 0],
                            [8, 13, 0], [9, 14, 0]], columns=['a', 'b', 'c'])
    assert_frame_equal(df1, res_int, check_index_type=False)
    print "etl_test - intersect - OK"

    print "\n|-------- Filter --------|\n"
    ddf_1 = DDF().parallelize(data, 4).filter('a > 5')
    df1 = ddf_1.cache().show()
    res_fil = pd.DataFrame([[6, 11, 0], [7, 12, 0],
                            [8, 13, 0], [9, 14, 0]], columns=['a', 'b', 'c'])
    assert_frame_equal(df1, res_fil, check_index_type=False)
    print "etl_test - filter - OK"

    print "\n|-------- Join --------|\n"
    ddf_1a = DDF().parallelize(data, 4)
    ddf_1b = DDF().parallelize(data, 4)
    ddf_2 = ddf_1a.join(ddf_1b, key1=['a'], key2=['a'])
    df1 = ddf_2.cache().show()
    res_join = pd.DataFrame([[0, 5, 0, 5, 0], [1, 6, 0, 6, 0],
                             [2, 7, 0, 7, 0], [3, 8, 0, 8, 0],
                             [4, 9, 0, 9, 0], [5, 10, 0, 10, 0],
                             [6, 11, 0, 11, 0], [7, 12, 0, 12, 0],
                             [8, 13, 0, 13, 0], [9, 14, 0, 14, 0]],
                            columns=['a', 'b_l', 'c_l', 'b_r', 'c_r'])
    assert_frame_equal(df1, res_join, check_index_type=False)
    print "etl_test - join - OK"

    print "\n|-------- Replace Values --------|\n"
    ddf_1 = DDF().parallelize(data, 4).replace({'c': [[0], [42]]})
    df1 = ddf_1.cache().show()
    res_rep = pd.DataFrame([[0, 5, 42], [1, 6, 42], [2, 7, 42], [4, 8, 42],
                            [5, 9, 42], [6, 10, 42], [6, 11, 42], [7, 12, 42],
                            [8, 13, 42], [9, 14, 42]], columns=['a', 'b', 'c'])
    assert_frame_equal(df1, res_rep, check_index_type=False)
    print "etl_test - replace - OK"

    print "\n|-------- Sample --------|\n"
    ddf_1 = DDF().parallelize(data, 4).sample(7)
    df1 = ddf_1.cache().show()
    if len(df1) != 7:
        raise Exception("Sample error")
    print "etl_test - sample - OK"

    print "\n|-------- Select --------|\n"
    ddf_1 = DDF().parallelize(data, 4).select(['a'])
    df1 = ddf_1.cache().show()
    res_rep = pd.DataFrame([[0], [1], [2], [3], [4],  [5], [6], [7],
                            [8], [9]], columns=['a'])
    assert_frame_equal(df1, res_rep, check_index_type=False)
    print "etl_test - select - OK"

    print "\n|-------- Sort --------|\n"
    ddf_1 = DDF().parallelize(data, 4).sort(['a'], ascending=[False])
    df1 = ddf_1.cache().show()
    res_sor = pd.DataFrame([[9, 14, 0], [8, 13, 0],  [7, 12, 0],
                            [6, 11, 0], [5, 10, 0], [4, 9, 0],
                            [3, 8, 0], [2, 7, 0], [1, 6, 0],
                            [0, 5, 0]], columns=['a', 'b', 'c'])
    assert_frame_equal(df1, res_sor, check_index_type=False)
    print "etl_test - sort - OK"

    print "\n|-------- Split --------|\n"
    ddf_1a, ddf_1b = DDF().parallelize(data, 4).split(0.5)
    df1 = ddf_1a.cache().show()
    df2 = ddf_1b.cache().show()
    cond = any(pd.concat([df1, df2]).duplicated(['a', 'b', 'c']).values)
    if cond:
        raise Exception("Split")
    print "etl_test - split - OK"

    print "\n|-------- Take --------|\n"
    ddf_1 = DDF().parallelize(data, 4).take(3)
    df1 = ddf_1.cache().show()
    res_tak = pd.DataFrame([[0, 5, 42], [1, 6, 42], [2, 7, 42]],
                           columns=['a', 'b', 'c'])
    assert_frame_equal(df1, res_tak, check_index_type=False)
    print "etl_test - take - OK"

    print "\n|-------- Map operation --------|\n"
    f = lambda col: 7 if col['a'] > 5 else col['a']
    ddf_1 = DDF().parallelize(data, 4).map(f, 'a')
    df1 = ddf_1.cache().show()
    res_tra = pd.DataFrame([[0, 5, 0], [1, 6, 0], [2, 7, 0], [3, 8, 0],
                            [4, 9, 0], [5, 10, 0], [7, 11, 0], [7, 12, 0],
                            [7, 13, 0], [7, 14, 0]], columns=['a', 'b', 'c'])
    assert_frame_equal(df1, res_tra, check_index_type=False)
    print "etl_test - transform - OK"

    print "\n|-------- Union --------|\n"
    ddf_1a = DDF().parallelize(data, 4)
    ddf_1b = DDF().parallelize(data1, 4)
    ddf_2 = ddf_1a.union(ddf_1b)
    df1 = ddf_2.cache().show()
    res_uni = pd.DataFrame([[0, 5, 0.0], [1, 6, 0.0], [2, 7, 0.0],
                            [0, 5, None], [1, 6, None], [3, 8, 0.0],
                            [4, 9, 0.0], [5, 10, 0.0], [2, 7, None],
                            [3, 8, None], [6, 11, 0.0], [7, 12, 0.0],
                            [8, 13, 0.0], [4, 9, None], [9, 14, 0.0]],
                           columns=['a', 'b', 'c'])
    assert_frame_equal(df1, res_uni, check_index_type=False)
    print "etl_test - union - OK"

    print "\n|-------- With_column --------|\n"
    ddf_1 = DDF().parallelize(data, 4).with_column('a', 'A')
    df1 = ddf_1.cache().show()
    res_with = pd.DataFrame([[0, 5, 0], [1, 6, 0], [2, 7, 0], [3, 8, 0],
                            [4, 9, 0], [5, 10, 0], [6, 11, 0], [7, 12, 0],
                            [8, 13, 0], [9, 14, 0]], columns=['A', 'b', 'c'])
    assert_frame_equal(df1, res_with, check_index_type=False)
    print "etl_test - with_column - OK"


if __name__ == '__main__':
    print "_____ETL_____"
    simple_etl()
