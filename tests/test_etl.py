#!/usr/bin/python
# -*- coding: utf-8 -*-

from ddf_library.ddf import DDF
import pandas as pd
import numpy as np


def etl():

    url = ('https://archive.ics.uci.edu/ml/'
           'machine-learning-databases/abalone/abalone.data')
    cols = ['sex', 'length', 'diam', 'height', 'weight', 'rings']
    data = pd.read_csv(url, usecols=[0, 1, 2, 3, 4, 8], names=cols)[:20]

    f1 = lambda col: -42 if col['height'] > 0.090 else col['height']
    data1 = DDF().parallelize(data, 4)\
                 .map(f1, 'height_nan').cache()

    data2 = data1.map(lambda col: "."+col['sex'], 'sex')\
                 .cache()

    print """
    ----------------------------------------------------------------------------
        etl_test_1: Multiple caches
        """

    df = data1.cache().show(5)
    print "etl_test_1a:", df
    print "Values on height_nan column: [-42, 0.09, -42, -42, 0.08]"

    df = data2.cache().show(5)
    print "etl_test_1b:", df
    print "Values on sex column: [.M, .M, .F, .M, .I]"

    df = data1.cache().show(5)
    print "etl_test_1c:", df
    print "Equals to etl_test_1a"

    print """
    ----------------------------------------------------------------------------
        etl_test_2: Branching:
        data2 e data3 são filhos de data1
        nesse caso: nenhum dos transforms das listas podem ser otimizadas
    """

    data3 = data1.drop(['length', 'diam']).cache()

    df1 = data1.show(5)
    df2 = data2.show(5)
    df3 = data3.show(5)

    print "etl_test_2a:", df1
    print "Values on sex column: [M, M, F, M, I]"
    print "etl_test_2b:", df2
    print "Values on sex column: [.M, .M, .F, .M, .I]"
    print "etl_test_2c:", df3
    print "Values on sex column: [M, M, F, M, I]"

    print """
    ----------------------------------------------------------------------------
          etl_test_3: Avaliar a capacidade de agrupar multiplas lazy tasks
          Nesse caso, drop, drop e replace vao ser agrupadas
    """
    data4 = data2.drop(['length']).drop(['diam'])\
        .replace({15: 42}, subset=['rings'])

    df = data4.cache().show(5)
    print "etl_test_3:", df

    print """
    ----------------------------------------------------------------------------
          etl_test_4: Avaliar a capacidade de produção de dois resultados
    """

    data5a, data5b = data4.split(0.5)

    df = data5b.cache().show(5)
    print "etl_test_4a:", df

    df = data5a.cache().show(5)
    print "etl_test_4b:", df
    print "4a and 4b must be differents"

    print """
    ----------------------------------------------------------------------------
        etl_test_5: Avaliar capacidade de esperar uma segunda entrada
    """

    data6 = data5b.join(data5a, ['rings'], ['rings'])\
        .filter('(rings > 8)')\
        .select(['sex_l', 'height_l', 'weight_l', 'rings'])\


    df = data6.cache().show()
    print "etl_test_5a len({}): {}".format(len(df), df[0:5])

    data7 = data6.sample(10).sort(['rings'], [True])
    data8 = data6.join(data7, ['rings'], ['rings'])

    df = data8.cache().show()
    print "etl_test_5b len({}): {}".format(len(df), df[0:5])
    print 'etl_test_5c:\n', data8.schema()
    print """
    ----------------------------------------------------------------------------
        etl_test_6: Avaliar capacidade de gerar resultado sem salvar varivel
    """
    df = data1.distinct(['rings']).show()
    df2 = data1.cache().show()
    print "etl_test_6a:", df
    print "etl_test_6b:", df2
    print 'Must be equals'

    print """
    ----------------------------------------------------------------------------
        etl_test_7: Avaliar capacidade 'count' and 'take'
    """

    v = data1.select(['rings']).count()
    df = data1.select(['rings']).take(7).show()

    print "etl_test_7a:", v
    print "etl_test_7b:", len(df)


def simple_etl():
    from pandas.util.testing import assert_frame_equal

    data = pd.DataFrame([[i, i+5, 0] for i in range(10)],
                        columns=['a', 'b', 'c'])

    data1 = pd.DataFrame([["Bob_{}".format(i), i + 5] for i in range(5)],
                         columns=['a', 'b'])

    data2 = pd.DataFrame([
        ["Alice_{}".format(i), i + 5, 0] for i in xrange(5, 15)],
                         columns=['a', 'b', 'c'])

    data3 = pd.DataFrame([[i, i + 5, 'hello'] for i in xrange(5, 15)],
                         columns=['a', 'b', 'c'])

    data3.loc[15, ['b']] = np.nan
    data3.loc[0:2, ['c']] = 'Hi'
    data3.loc[6:9, ['c']] = 'world'
    data3['d'] = [10,  12, 13, 19, 19,19,19,19,19, 19, np.nan]
    data3['e'] = [10,  10, 10, 10, 10, 10, 10, 10, 10, 10, np.nan]

    data3['f'] = [10, 10, 10, 10, 10, 10, 10, 10, 10, np.nan, np.nan]
    data3['g'] = [10, 12, 13, 19, 19, 19, 19, 19, 19, np.nan, np.nan]

    data3['h'] = [10, 12, 13, 19, 5, 5, 5, 5, 5, np.nan, np.nan]
    data3['i'] = [5, 12, 13, 19, 19, 19, 5, 5, 5, 5, np.nan]

    # print "\n|-------- Read Data --------|\n"
    # ddf_1 = DDF().load_text('/test-read_data.csv', header=True,
    #                         sep=',', dtype={'la1el': np.dtype('O'),
    #                                         'x': np.float64,
    #                                         'y': np.float64}).select(['x', 'y'])

    #
    # print "Schema: \n", ddf_1.schema()
    # return 0

    # print "\n|-------- Add Column --------|\n"
    # ddf_1a = DDF().parallelize(data1, 5) # 2
    # ddf_1b = DDF().parallelize(data2, 10)
    # df1 = ddf_1a.add_column(ddf_1b).show()
    #
    # res_add = pd.DataFrame([[0,     5,     5, 10,  0],
    #                         [1,     6,     6, 11,  0],
    #                         [2,     7,     7, 12,  0],
    #                         [3,     8,     8, 13,  0],
    #                         [4,     9,     9, 14,  0],
    #                         [None, 10,  None, 15,  0],
    #                         [None, 11,  None, 16,  0],
    #                         [None, 12,  None, 17,  0],
    #                         [None, 13,  None, 18,  0],
    #                         [None, 14,  None, 19,  0]
    #                         ], columns=['a_l', 'a_r', 'b_l', 'b_r', 'c'])
    # print df1.equals(res_add)
    # assert_frame_equal(df1, res_add, check_index_type=False) #, check_dtype=False)
    # print df1

    #
    # print "etl_test - add column - OK",
    #
    # print "\n|-------- Aggregation --------|\n"
    # express = {'a': ['count'], 'b': ['first', 'last']}
    # # ddf_1 = DDF().parallelize(data, 4).group_by(['c']).agg(express)
    # ddf_1 = DDF().parallelize(data3, 4).group_by(['a', 'c']).count('*')
    #
    # # .aggregation(['c'], exprs=express, aliases=aliases)
    # df1 = ddf_1.cache().show()
    # print df1
    # # res_agg = pd.DataFrame([[0, 10, 5, 14]],
    #                        columns=['c', 'COUNT', 'col_First', 'col_Last'])
    # # assert_frame_equal(df1, res_agg, check_index_type=False)
    # print "etl_test - aggregation - OK",

    print "\n|-------- CrossTab --------|\n"
    data = pd.DataFrame([(1, 1), (1, 2), (2, 1), (2, 1),
                         (2, 3), (3, 2), (3, 3)], columns=['key', 'value'])
    ddf_1 = DDF().parallelize(data, 4).cross_tab(col1='key', col2='value')
    df1 = ddf_1.show()
    print df1
    """
    +---------+---+---+---+
    |key_value|  1|  2|  3|
    +---------+---+---+---+
    |        2|  2|  0|  1|
    |        1|  1|  1|  0|
    |        3|  0|  1|  1|
    +---------+---+---+---+
    """

    # print "\n|-------- DropNaN --------|\n"
    # ddf_1 = DDF().parallelize(data3, 4)
    # df1a = ddf_1.dropna(['c'], mode='REMOVE_COLUMN', how='all', thresh=1)
    #
    # df1b = ddf_1.dropna(['c'], mode='REMOVE_ROW', how='any')
    # print df1a.show()
    # print df1b.show()
    # return 0


    # print "\n|-------- FillNaN --------|\n"
    # print data3
    # ddf_1 = DDF().parallelize(data3, 4)
    # # df1a = ddf_1.fillna(mode='VALUE', value=42),
    # # df1a = ddf_1.fillna(mode='VALUE', value={'c': 42})
    # # df1a = ddf_1.fillna(['a', 'b'], mode='MEAN')
    # # df1a = ddf_1.fillna(['c'], mode='MODE')
    #
    # df1a = ddf_1.fillna(['a', 'b', 'd','e','f','g', 'h', 'i'], mode='MEDIAN')
    #
    # print df1a.show()
    # print "A: 9.5 - B: 14.5 - D: 19.0 - E: 10.0 - G: 19.0 - H: 5.0 - I: 8.5"




    # print "\n|-------- CrossJoin --------|\n"
    # ddf_1a = DDF().parallelize(data1, 4)
    # ddf_1b = DDF().parallelize(data2, 4)
    # ddf_2 = ddf_1a.cross_join(ddf_1b)
    # df1 = ddf_2.cache().show(50)
    #
    #
    # print "\n|-------- Describe --------|\n"
    # df1 = DDF().parallelize(data3, 4).describe()
    # print df1
    # print "etl_test - difference - OK"

    # print "\n|-------- Covariance --------|\n"
    # # df = pd.DataFrame([[1692, 68], [1978, 102],
    # #                    [1884, 110], [2151, 112],
    # #                    [2519, 154]], columns=['a', 'b'])
    # # cov(a,b) = 9107.3
    #
    # df = pd.DataFrame([[1.95, 93.1], [1.96, 93.9], [1.95, 89.9],
    #                    [1.98, 95.1], [2.10, 100.2]], columns=['a', 'b'])
    # # cov(a,b) = 0.2196
    #
    # df1 = DDF().parallelize(df, 4).covariance(col1='a', col2='b')
    # print df1
    # print "etl_test - Covariance - OK"

    # print "\n|-------- Correlation --------|\n"
    #
    # df = pd.DataFrame([[8, 81], [8, 80], [6, 75],
    #                    [5, 65], [7, 91], [6, 80]], columns=['a', 'b'])
    # # corr(a,b) = 0.6475106
    #
    # df1 = DDF().parallelize(df, 4).correlation(col1='a', col2='b')
    # print df1
    # print "etl_test - Correlation - OK"

    # print "\n|-------- Subtract --------|\n"
    # s1 = pd.DataFrame([("a", 1), ("a", 1), ("a", 1), ("a", 2), ("b",  3),
    #                    ("c", 4)], columns=['col1', 'col2'])
    # s2 = pd.DataFrame([("a", 1), ("b",  3)], columns=['col1', 'col2'])
    # ddf_1a = DDF().parallelize(s1, 4)
    # ddf_1b = DDF().parallelize(s2, 4)
    # ddf_2 = ddf_1a.subtract(ddf_1b)
    # df1 = ddf_2.cache().show()
    # print df1
    # print "etl_test - subtract - OK"

    # print "\n|-------- ExceptAll --------|\n"
    # s1 = pd.DataFrame([("a", 1), ("a", 1), ("a", 1), ("a", 2), ("b",  3),
    #                    ("c", 4)], columns=['col1', 'col2'])
    # s2 = pd.DataFrame([("a", 1), ("b",  3)], columns=['col1', 'col2'])
    # ddf_1a = DDF().parallelize(s1, 4)
    # ddf_1b = DDF().parallelize(s2, 4)
    # ddf_2 = ddf_1a.subtract(ddf_1b)
    # df1 = ddf_2.cache().show()
    # print df1
    # print "etl_test - exceptAll - OK"

    # print "\n|-------- Distinct --------|\n"
    # ddf_1 = DDF().parallelize(data, 4).distinct(['c'])
    # df1 = ddf_1.cache().show()
    #
    # res_dist = pd.DataFrame([[0, 5, 0]], columns=['a', 'b', 'c'])
    # assert_frame_equal(df1, res_dist, check_index_type=False)
    # print "etl_test - distinct - OK"
    #
    # print "\n|-------- Drop --------|\n"
    # ddf_1 = DDF().parallelize(data, 4).drop(['a'])
    # df1 = ddf_1.cache().show()
    # res_drop = pd.DataFrame([[5, 0], [6, 0], [7, 0], [8, 0], [9, 0],
    #                          [10, 0], [11, 0], [12, 0],
    #                          [13, 0], [14, 0]], columns=['b', 'c'])
    # assert_frame_equal(df1, res_drop, check_index_type=False)
    # print "etl_test - drop - OK"
    #
    # print "\n|-------- Intersect All--------|\n"
    # s1 = pd.DataFrame([("a", 1), ("a", 1), ("a", 1), ("a", 2), ("b", 3),
    #                    ("c", 4)], columns=['col1', 'col2'])
    # s2 = pd.DataFrame([('a', 1), ('a', 1), ('b', 3)], columns=['col1', 'col2'])
    #
    # ddf_1a = DDF().parallelize(s1, 4)
    # ddf_1b = DDF().parallelize(s2, 4)
    # ddf_2 = ddf_1a.intersect_all(ddf_1b)
    # df1 = ddf_2.cache().show()
    # print df1
    # print "etl_test - intersect - OK"
    #
    # print "\n|-------- Intersect --------|\n"
    # s1 = pd.DataFrame([("a", 1), ("a", 1), ("a", 1), ("a", 2), ("b", 3),
    #                    ("c", 4)], columns=['col1', 'col2'])
    # s2 = pd.DataFrame([('a', 1), ('a', 1), ('b', 3)], columns=['col1', 'col2'])
    #
    # ddf_1a = DDF().parallelize(s1, 4)
    # ddf_1b = DDF().parallelize(s2, 4)
    # ddf_2 = ddf_1a.intersect(ddf_1b)
    # df1 = ddf_2.cache().show()
    # print df1
    # print "etl_test - intersect - OK"
    #
    # print "\n|-------- Filter --------|\n"
    # ddf_1 = DDF().parallelize(data, 4).filter('a > 5')
    # df1 = ddf_1.cache().show()
    # res_fil = pd.DataFrame([[6, 11, 0], [7, 12, 0],
    #                         [8, 13, 0], [9, 14, 0]], columns=['a', 'b', 'c'])
    # assert_frame_equal(df1, res_fil, check_index_type=False)
    # print "etl_test - filter - OK"
    #
    # print "\n|-------- Join --------|\n"
    # ddf_1a = DDF().parallelize(data, 4)
    # ddf_1b = DDF().parallelize(data, 4)
    # ddf_2 = ddf_1a.join(ddf_1b, key1=['a'], key2=['a'])
    # df1 = ddf_2.cache().show()
    # res_join = pd.DataFrame([[0, 5, 0, 5, 0], [1, 6, 0, 6, 0],
    #                          [2, 7, 0, 7, 0], [3, 8, 0, 8, 0],
    #                          [4, 9, 0, 9, 0], [5, 10, 0, 10, 0],
    #                          [6, 11, 0, 11, 0], [7, 12, 0, 12, 0],
    #                          [8, 13, 0, 13, 0], [9, 14, 0, 14, 0]],
    #                         columns=['a', 'b_l', 'c_l', 'b_r', 'c_r'])
    # assert_frame_equal(df1, res_join, check_index_type=False)
    # print "etl_test - join - OK"
    #
    # print "\n|-------- Replace Values --------|\n"
    # ddf_1 = DDF().parallelize(data, 4).replace({'c': [[0], [42]]})
    # df1 = ddf_1.cache().show()
    # res_rep = pd.DataFrame([[0, 5, 42], [1, 6, 42], [2, 7, 42], [4, 8, 42],
    #                         [5, 9, 42], [6, 10, 42], [6, 11, 42], [7, 12, 42],
    #                         [8, 13, 42], [9, 14, 42]], columns=['a', 'b', 'c'])
    # assert_frame_equal(df1, res_rep, check_index_type=False)
    # print "etl_test - replace - OK"
    #
    # print "\n|-------- Sample --------|\n"
    # ddf_1 = DDF().parallelize(data, 4).sample(7)
    # df1 = ddf_1.cache().show()
    # if len(df1) != 7:
    #     raise Exception("Sample error")
    # print "etl_test - sample - OK"

    # print "\n|-------- Select --------|\n"
    # ddf_1 = DDF().parallelize(data, 4).select(['a'])
    # df1 = ddf_1.cache().show()
    # res_rep = pd.DataFrame([[0], [1], [2], [3], [4],  [5], [6], [7],
    #                         [8], [9]], columns=['a'])
    # assert_frame_equal(df1, res_rep, check_index_type=False)
    # print "etl_test - select - OK"
    #
    # print "\n|-------- Sort --------|\n"
    # ddf_1 = DDF().parallelize(data, 4).sort(['a'], ascending=[False])
    # df1 = ddf_1.cache().show()
    # res_sor = pd.DataFrame([[9, 14, 0], [8, 13, 0],  [7, 12, 0],
    #                         [6, 11, 0], [5, 10, 0], [4, 9, 0],
    #                         [3, 8, 0], [2, 7, 0], [1, 6, 0],
    #                         [0, 5, 0]], columns=['a', 'b', 'c'])
    # assert_frame_equal(df1, res_sor, check_index_type=False)
    # print "etl_test - sort - OK"
    #
    # print "\n|-------- Split --------|\n"
    # ddf_1a, ddf_1b = DDF().parallelize(data, 4).split(0.5)
    # df1 = ddf_1a.cache().show()
    # df2 = ddf_1b.cache().show()
    # cond = any(pd.concat([df1, df2]).duplicated(['a', 'b', 'c']).values)
    # if cond:
    #     raise Exception("Split")
    # print "etl_test - split - OK"

    # print "\n|-------- Take --------|\n"
    # ddf_1 = DDF().parallelize(data, 4).take(3)
    # df1 = ddf_1.cache().show()
    # res_tak = pd.DataFrame([[0, 5, 42], [1, 6, 42], [2, 7, 42]],
    #                        columns=['a', 'b', 'c'])
    # assert_frame_equal(df1, res_tak, check_index_type=False)
    # print "etl_test - take - OK"

    # print "\n|-------- Map operation --------|\n"
    # f = lambda col: 7 if col['a'] > 5 else col['a']
    # ddf_1 = DDF().parallelize(data, 4).map(f, 'a')
    # df1 = ddf_1.cache().show()
    # res_tra = pd.DataFrame([[0, 5, 0], [1, 6, 0], [2, 7, 0], [3, 8, 0],
    #                         [4, 9, 0], [5, 10, 0], [7, 11, 0], [7, 12, 0],
    #                         [7, 13, 0], [7, 14, 0]], columns=['a', 'b', 'c'])
    # assert_frame_equal(df1, res_tra, check_index_type=False)
    # print "etl_test - transform - OK"
    #
    # print "\n|-------- Union by Name --------|\n"
    # data = pd.DataFrame([[i, 5] for i in range(10)], columns=['a', 'b'])
    # data1 = pd.DataFrame([["i{}".format(i), 7] for i in range(5)],
    #                      columns=['b', 'a'])
    #
    # ddf_1a = DDF().parallelize(data, 4)
    # ddf_1b = DDF().parallelize(data1, 4)
    # ddf_2 = ddf_1a.union_by_name(ddf_1b)
    # df1 = ddf_2.cache().show()
    # print df1
    #
    # print "etl_test - union by name - OK"
    #
    # print "\n|-------- Union --------|\n"
    # data = pd.DataFrame([[i, 5, 10] for i in range(10)], columns=['a', 'b', 'c'])
    # data1 = pd.DataFrame([["i{}".format(i), 7] for i in range(5)],
    #                      columns=['b', 'a'])
    #
    # ddf_1a = DDF().parallelize(data, 4)
    # ddf_1b = DDF().parallelize(data1, 4)
    # ddf_2 = ddf_1a.union(ddf_1b)
    # df1 = ddf_2.cache().show()
    # print df1
    #
    # print "etl_test - union - OK"

    #
    # print "\n|-------- cast --------|\n"
    # ddf_1 = DDF().parallelize(data, 4).cast(['a', 'b'], 'string')
    # schema = ddf_1.cache().schema()
    # print schema
    # print "etl_test - with_column - OK"

    # print "\n|-------- With_column Renamed --------|\n"
    # ddf_1 = DDF().parallelize(data, 4).with_column_renamed('a', 'A')
    # df1 = ddf_1.cache().show()
    # res_with = pd.DataFrame([[0, 5, 0], [1, 6, 0], [2, 7, 0], [3, 8, 0],
    #                         [4, 9, 0], [5, 10, 0], [6, 11, 0], [7, 12, 0],
    #                         [8, 13, 0], [9, 14, 0]], columns=['A', 'b', 'c'])
    # assert_frame_equal(df1, res_with, check_index_type=False)
    # print "etl_test - with_column - OK"


if __name__ == '__main__':
    print "_____ETL_____"
    simple_etl()
    #etl()
