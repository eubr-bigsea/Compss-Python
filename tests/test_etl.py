#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pycompss.api.task import task
from ddf_library.ddf import DDF
from ddf_library.utils import generate_data
import pandas as pd
import numpy as np
import time
from pandas.util.testing import assert_frame_equal


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

    print("""
    ----------------------------------------------------------------------------
        etl_test_1: Multiple caches
        """)

    df = data1.cache().to_df()
    print("etl_test_1a:", df[:5])
    print("Values on height_nan column: [-42, 0.09, -42, -42, 0.08]")

    df = data2.cache().to_df()
    print("etl_test_1b:", df[:5])
    print("Values on sex column: [.M, .M, .F, .M, .I]")

    df = data1.cache().to_df()
    print("etl_test_1c:", df[:5])
    print("Equals to etl_test_1a")

    print("""
    ----------------------------------------------------------------------------
        etl_test_2: Branching:
        data2 e data3 são filhos de data1
        nesse caso: nenhum dos transforms das listas podem ser otimizadas
    """)

    data3 = data1.drop(['length', 'diam']).cache()

    df1 = data1.to_df()
    df2 = data2.to_df()
    df3 = data3.to_df()

    print("etl_test_2a:", df1[0:5])
    print("Values on sex column: [M, M, F, M, I]")
    print("etl_test_2b:", df2[0:5])
    print("Values on sex column: [.M, .M, .F, .M, .I]")
    print("etl_test_2c:", df3[0:5])
    print("Values on sex column: [M, M, F, M, I]")

    print("""
    ----------------------------------------------------------------------------
          etl_test_3: Avaliar a capacidade de agrupar multiplas lazy tasks
          Nesse caso, drop, drop e replace vao ser agrupadas
    """)
    data4 = data2.drop(['length']).drop(['diam'])\
        .replace({15: 42}, subset=['rings'])

    df = data4.to_df()[0:5]
    print("etl_test_3:", df)

    print("""
    ----------------------------------------------------------------------------
          etl_test_4: Avaliar a capacidade de produção de dois resultados
    """)

    data5a, data5b = data4.split(0.5)

    df = data5b.to_df()
    print("etl_test_4a:", df[:5])

    df = data5a.to_df()
    print("etl_test_4b:", df[:5])
    print("4a and 4b must be differents")

    print("""
    ----------------------------------------------------------------------------
        etl_test_5: Avaliar capacidade de esperar uma segunda entrada
    """)

    data6 = data5b.join(data5a, ['rings'], ['rings'])\
        .filter('(rings > 8)')\
        .select(['sex_l', 'height_l', 'weight_l', 'rings'])\


    df = data6.to_df()
    print("etl_test_5a len({}): {}".format(len(df), df[0:5]))

    data7 = data6.sample(7).sort(['rings'], [True])
    data8 = data6.join(data7, ['rings'], ['rings'])

    df = data8.to_df()
    print("etl_test_5b len({}): {}".format(len(df), df[0:5]))
    print('etl_test_5c:\n', data8.schema())
    print("""
    ----------------------------------------------------------------------------
        etl_test_6: Avaliar capacidade de gerar resultado sem salvar varivel
    """)
    df = data1.distinct(['rings']).to_df()
    df2 = data1.cache().to_df()
    print("etl_test_6a:", df)
    print("etl_test_6b:", df2)
    print('Must be equals')

    print("""
    ----------------------------------------------------------------------------
        etl_test_7: Avaliar capacidade 'count' and 'take'
    """)

    v = data1.select(['rings']).count_rows()
    len_df = data1.select(['rings']).take(7).count_rows()

    print("etl_test_7a:", v)
    print("etl_test_7b:", len_df)


def add_columns():
    print("\n|-------- Add Column --------|\n")
    data1 = pd.DataFrame([["N_{}".format(i)] for i in range(5)],
                         columns=['name'])

    data2 = pd.DataFrame([["A_{}".format(i), i + 5] for i in range(5, 15)],
                         columns=['name', 'b'])

    ddf_1a = DDF().parallelize(data1, 5)
    ddf_1b = DDF().parallelize(data2, 10)
    df1 = ddf_1a.add_column(ddf_1b).to_df()

    res = pd.merge(data1, data2, left_index=True,  right_index=True,
                   suffixes=['_l', '_r'])

    assert_frame_equal(df1, res, check_index_type=False)

    print("etl_test - add column - OK")


def aggregation():
    print("\n|-------- Aggregation --------|\n")
    n = 10
    data3 = pd.DataFrame([[i, i + 5, 'hello'] for i in range(n)],
                         columns=['a', 'b', 'c'])

    express = {'a': ['count'], 'b': ['first', 'last']}
    ddf_1 = DDF().parallelize(data3, 4).group_by(['c']).agg(express)
    df = ddf_1.to_df()
    cond1 = len(df) == 1
    cond2 = all([f == n for f in df['count(a)'].values])
    if not (cond1 and cond2):
        print(df)
        raise Exception('Error in aggregation')

    ddf_1 = DDF().parallelize(data3, 4).group_by(['a', 'c']).count('*')
    df = ddf_1.to_df()
    cond1 = len(df) == n
    cond2 = all([f == 1 for f in df['count(*)'].values])
    if not (cond1 and cond2):
        print(df)
        raise Exception('Error in aggregation')

    print("etl_test - aggregation - OK")


def balancer():
    print("\n|-------- Balance --------|\n")

    iterations = [[10, 0, 10, 5, 100],
                  [100, 5, 10, 0, 10],
                  [85, 0, 32, 0, 0],
                  [0, 0, 0, 30, 100]
                  ]

    for s in iterations:
        print('Before:', s)
        data, info = generate_data(s)
        ddf_1 = DDF().import_data(data, info)
        df1 = ddf_1.to_df()['a'].values

        ddf_2 = ddf_1.balancer(forced=True).cache()
        size_a = ddf_2.count_rows(total=False)
        df2 = ddf_1.to_df()['a'].values

        print('After:', size_a)
        print(np.array_equal(df1, df2))


def cast():
    print("\n|-------- cast --------|\n")
    data = pd.DataFrame([[i, i + 5, 0] for i in range(10)],
                        columns=['a', 'b', 'c'])
    ddf_1 = DDF().parallelize(data, 4).cast(['a', 'b'], 'string')
    schema = ddf_1.schema()
    print(schema)
    print("etl_test - cast - OK")


def cross_join():
    print("\n|-------- CrossJoin --------|\n")
    data1 = pd.DataFrame([["Bob_{}".format(i), i + 5] for i in range(5)],
                         columns=['name', 'height'])
    data2 = pd.DataFrame([[i + 5] for i in range(5, 15)], columns=['gain'])

    ddf_1a = DDF().parallelize(data1, 4)
    ddf_1b = DDF().parallelize(data2, 4)
    df1 = ddf_1a.cross_join(ddf_1b).to_df().sort_values(by=['name', 'gain'])
    print(df1[0:50])


def distinct():
    print("\n|-------- Distinct --------|\n")
    data = pd.DataFrame([[i, i + 5, 0] for i in range(10)],
                        columns=['a', 'b', 'c'])

    ddf_1 = DDF().parallelize(data, 4).distinct(['c'])
    df1 = ddf_1.cache().to_df()
    print(df1)
    res_dist = pd.DataFrame([[0, 5, 0]], columns=['a', 'b', 'c'])
    assert_frame_equal(df1, res_dist, check_index_type=False)
    print("etl_test - distinct - OK")


def drop():
    print("\n|-------- Drop --------|\n")
    data = pd.DataFrame([[i, i + 5, 0] for i in range(10)],
                        columns=['a', 'b', 'c'])

    ddf_1 = DDF().parallelize(data, 4).drop(['a'])
    df1 = ddf_1.to_df()
    res_drop = pd.DataFrame([[5, 0], [6, 0], [7, 0], [8, 0], [9, 0],
                             [10, 0], [11, 0], [12, 0],
                             [13, 0], [14, 0]], columns=['b', 'c'])
    assert_frame_equal(df1, res_drop, check_index_type=False)
    print("etl_test - drop - OK")


def drop_na():
    print("\n|-------- DropNaN --------|\n")
    data3 = pd.DataFrame([[i, i + 5, 'hello'] for i in range(5, 15)],
                         columns=['a', 'b', 'c'])

    data3.loc[15, ['b']] = np.nan
    data3['d'] = [10, 12, 13, 19, 19, 19, 19, 19, 19, 19, np.nan]
    data3['g'] = [10, 12, 13, 19, 19, 19, 19, 19, 19, np.nan, np.nan]

    ddf_1 = DDF().parallelize(data3, 4)
    df1a = ddf_1.dropna(['c'], mode='REMOVE_COLUMN', how='all', thresh=1)
    df1b = ddf_1.dropna(['c'], mode='REMOVE_ROW', how='any')

    print(df1a.to_df())
    print(df1b.to_df())


def except_all():
    print("\n|-------- ExceptAll --------|\n")
    cols = ['a', 'b']
    s1 = pd.DataFrame([("a", 1), ("a", 1), ("a", 1), ("a", 2), ("b",  3),
                       ("c", 4)], columns=cols)
    s2 = pd.DataFrame([("a", 1), ("b",  3), ('e', 4), ('e', 4), ('e', 4),
                       ('e', 6), ('e', 9), ('e', 10), ('e', 4), ('e', 4)],
                      columns=cols)

    ddf_1a = DDF().parallelize(s1, 2)
    ddf_1b = DDF().parallelize(s2, 4)
    ddf_2 = ddf_1a.except_all(ddf_1b)
    df1 = ddf_2.to_df()
    print(df1)

    """
    ("a", 1),
    ("a", 1),
    ("a", 2),
    ("c", 4)
    """
    res = pd.DataFrame([("a", 1), ("a", 1), ("a", 2), ("c",  4)], columns=cols)
    assert_frame_equal(df1, res, check_index_type=False)
    print("etl_test - exceptAll - OK")


def explode():
    print("\n|-------- Explode --------|\n")

    df_size = 1 * 10 ** 3

    df = pd.DataFrame(np.random.randint(1, df_size, (df_size, 2)),
                      columns=list("AB"))
    df['C'] = df[['A', 'B']].values.tolist()

    col = 'C'

    ddf1 = DDF().parallelize(df, 4).explode(col)
    ddf1.show()
    print("etl_test - explode - OK")


def filter_operation():
    print("\n|-------- Filter --------|\n")
    data = pd.DataFrame([[i, i + 5] for i in range(10)], columns=['a', 'b'])

    ddf_1 = DDF().parallelize(data, 4).filter('a > 5')
    df1 = ddf_1.to_df()
    res_fil = pd.DataFrame([[6, 11], [7, 12],
                            [8, 13], [9, 14]], columns=['a', 'b'])
    assert_frame_equal(df1, res_fil, check_index_type=False)
    print("etl_test - filter - OK")


def fill_na():
    print("\n|-------- FillNaN --------|\n")
    data3 = pd.DataFrame([[i, i + 5, 'hello'] for i in range(5, 15)],
                         columns=['a', 'b', 'c'])
    data3.loc[15, ['b']] = np.nan
    data3.loc[0:2, ['c']] = 'Hi'
    data3.loc[6:9, ['c']] = 'world'
    data3['d'] = [10, 12, 13, 19, 19, 19, 19, 19, 19, 19, np.nan]
    data3['e'] = [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, np.nan]
    data3['f'] = [10, 10, 10, 10, 10, 10, 10, 10, 10, np.nan, np.nan]
    data3['g'] = [10, 12, 13, 19, 19, 19, 19, 19, 19, np.nan, np.nan]
    data3['h'] = [10, 12, 13, 19, 5, 5, 5, 5, 5, np.nan, np.nan]
    data3['i'] = [5, 12, 13, 19, 19, 19, 5, 5, 5, 5, np.nan]

    ddf_1 = DDF().parallelize(data3, 4)
    # df1a = ddf_1.fillna(mode='VALUE', value=42),
    # df1a = ddf_1.fillna(mode='VALUE', value={'c': 42})
    # df1a = ddf_1.fillna(['a', 'b'], mode='MEAN')
    # df1a = ddf_1.fillna(['c'], mode='MODE')

    df1a = ddf_1.fillna(['a', 'b', 'd','e','f','g', 'h', 'i'], mode='MEDIAN')

    print(df1a.to_df())
    print("A: 9.5 - B: 14.5 - D: 19.0 - E: 10.0 - G: 19.0 - H: 5.0 - I: 8.5")


def hash_partition():
    print("\n|-------- Hash partition --------|\n")
    n_rows = 1000
    data = pd.DataFrame({'a': np.random.randint(0, 100000, size=n_rows),
                         'b': np.random.randint(0, 100000, size=n_rows),
                         'c': np.random.randint(0, 100000, size=n_rows)
                         })
    data['b'] = data['b'].astype(str)

    ddf_1 = DDF().parallelize(data, 12).hash_partition(columns=['a', 'b'],
                                                       nfrag=6)
    f = ddf_1.num_of_partitions()
    c = ddf_1.count_rows(total=False)
    print(ddf_1.count_rows(total=False))
    print(sum(c) == n_rows)
    print(f == 6)
    df1 = ddf_1.to_df().sort_values(by=['a', 'b'])
    data = data.sort_values(by=['a', 'b'])
    assert_frame_equal(df1, data, check_index_type=False)
    print("etl_test - hash_partition - OK")


def import_data():
    print("\n|-------- Import data --------|\n")
    s1 = pd.DataFrame([("a", 1), ("a", 1), ("a", 1), ("a", 2), ("b", 3),
                       ("c", 4)], columns=['col1', 'col2'])

    df1 = DDF().import_data(np.array_split(s1, 4)).to_df()
    assert_frame_equal(df1, s1, check_index_type=False)
    print("etl_test - import data - OK")


def intersect():
    print("\n|-------- Intersect --------|\n")
    cols = ['col1', 'col2']
    s1 = pd.DataFrame([("a", 1), ("a", 1), ("a", 1), ("a", 2), ("b", 3),
                       ("c", 4)], columns=cols)
    s2 = pd.DataFrame([('a', 1), ('a', 1), ('b', 3)], columns=cols)

    ddf_1a = DDF().parallelize(s1, 4)
    ddf_1b = DDF().parallelize(s2, 4)
    ddf_2 = ddf_1a.intersect(ddf_1b)

    df1 = ddf_2.to_df().sort_values(by=cols)
    res = pd.DataFrame([['b', 3], ['a', 1]], columns=cols)
    res.sort_values(by=cols, inplace=True)

    assert_frame_equal(df1, res, check_index_type=False)
    print("etl_test - intersect - OK")


def intersect_all():
    print("\n|-------- Intersect All--------|\n")
    cols = ['col1', 'col2']
    s1 = pd.DataFrame([('a', 1), ('a', 1), ('b', 3), ('c', 4)], columns=cols)
    s2 = pd.DataFrame([('a', 1), ('a', 1), ('b', 3)], columns=cols)

    ddf_1a = DDF().parallelize(s1, 4)
    ddf_1b = DDF().parallelize(s2, 4)
    ddf_2 = ddf_1a.intersect_all(ddf_1b)

    df1 = ddf_2.to_df().sort_values(by=cols)
    res = pd.DataFrame([['b', 3], ['a', 1], ['a', 1]], columns=cols)
    res.sort_values(by=cols, inplace=True)

    assert_frame_equal(df1, res, check_index_type=False)
    print("etl_test - intersect all - OK")


def join():
    print("\n|--------  inner join --------|\n")

    data1 = pd.DataFrame([[i, i + 5, 0] for i in range(10)],
                         columns=['a', 'b', 'c'])
    data2 = data1.copy()
    data2.sample(frac=1,  replace=False)

    ddf_1a = DDF().parallelize(data1, 4)
    ddf_1b = DDF().parallelize(data2, 4)
    ddf_2 = ddf_1a.join(ddf_1b, key1=['a'], key2=['a'], case=False)
    df1 = ddf_2.to_df().sort_values(by=['a'])
    print(df1)

    print("etl_test - inner join - OK")

    print("\n|--------  left join --------|\n")
    data1 = pd.DataFrame([[i, i + 5, 0] for i in range(100)],
                         columns=['a', 'b', 'c'])
    data2 = data1.copy()[0:50]
    data2.sample(frac=1,  replace=False)
    data2.drop(['b', 'c'], axis=1, inplace=True)
    data2['d'] = 'd'

    ddf_1a = DDF().parallelize(data1, 4)
    ddf_1b = DDF().parallelize(data2, 4)
    ddf_2 = ddf_1a.join(ddf_1b, key1=['a'], key2=['a'], mode='left')
    df1 = ddf_2.to_df().sort_values(by=['a'])
    print(df1)

    print("etl_test - left join - OK")

    print("\n|--------  right join --------|\n")
    data1 = pd.DataFrame([[i, i + 5, 0] for i in range(100)],
                         columns=['a', 'b', 'c'])
    data1['b'] = data1['b'].astype('Int8')
    data1['c'] = data1['c'].astype('Int8')

    data2 = data1.copy()
    data1 = data1[0:50]
    data2.sample(frac=1,  replace=False)
    data2.drop(['b', 'c'], axis=1, inplace=True)
    data2['d'] = 'd'

    ddf_1a = DDF().parallelize(data1, 4)
    ddf_1b = DDF().parallelize(data2, 4)
    ddf_2 = ddf_1a.join(ddf_1b, key1=['a'], key2=['a'], mode='right')
    df1 = ddf_2.to_df().sort_values(by=['a'])
    print(df1)

    print("etl_test - right join - OK")


def map():
    print("\n|-------- Map operation --------|\n")
    data = pd.DataFrame([[i, i + 5, 0] for i in range(10)],
                        columns=['a', 'b', 'c'])

    def f(col):
        return 7 if col['a'] > 5 else col['a']

    ddf_1 = DDF().parallelize(data, 4).map(f, 'a')
    df1 = ddf_1.to_df()
    res_tra = pd.DataFrame([[0, 5, 0], [1, 6, 0], [2, 7, 0], [3, 8, 0],
                            [4, 9, 0], [5, 10, 0], [7, 11, 0], [7, 12, 0],
                            [7, 13, 0], [7, 14, 0]], columns=['a', 'b', 'c'])
    assert_frame_equal(df1, res_tra, check_index_type=False)
    print("etl_test - map - OK")


def read_data_single_fs():
    print("\n|-------- Read Data from a single file on FS --------|\n")
    dtypes = {'sepal_length': np.float64, 'sepal_width': np.float64,
              'petal_length': np.float64, 'petal_width': np.float64,
              'class': np.dtype('O')}
    ddf_1 = DDF().load_text('./iris-dataset.csv', header=True, storage='fs',
                            sep=',', dtype=dtypes, distributed=False)\
        .select(['class', 'sepal_length'])

    print(ddf_1.schema())
    print("Number of partitions: ", ddf_1.num_of_partitions())
    print("Number of rows: ", ddf_1.count_rows())


def read_data_multi_fs():
    print("\n|-------- Read Data from files in a folder on FS --------|\n")
    dtypes = {'sepal_length': np.float64, 'sepal_width': np.float64,
              'petal_length': np.float64, 'petal_width': np.float64,
              'class': np.dtype('O')}
    ddf_1 = DDF().load_text('~/iris_dataset_folder/', header=True, storage='fs',
                            sep=',', dtype=dtypes, distributed=True)\
        .select(['class', 'sepal_width'])

    print(ddf_1.schema())
    print("Number of partitions: ", ddf_1.num_of_partitions())
    print("Number of rows: ", ddf_1.count_rows())


def read_data_single_hdfs():
    print("\n|-------- Read Data From a single file on HDFS --------|\n")
    dtypes = {'la1el': np.dtype('O'), 'x': np.float64, 'y': np.float64}
    ddf_1 = DDF().load_text('/test-read_data.csv', header=True, storage='hdfs',
                            sep=',', dtype=dtypes,
                            distributed=False).select(['x', 'y'])

    print(ddf_1.schema())
    print("Number of partitions: ", ddf_1.num_of_partitions())
    print("Number of rows: ", ddf_1.count_rows())


def read_data_multi_hdfs():
    print("\n|-------- Read Data from files in a folder on HDFS --------|\n")
    dtypes = {'sepal_length': np.float64, 'sepal_width': np.float64,
              'petal_length': np.float64, 'petal_width': np.float64,
              'class': np.dtype('O')}
    ddf_1 = DDF().load_text('/iris_dataset_folder/', header=True,
                            storage='hdfs', sep=',', dtype=dtypes,
                            distributed=True)\
        .select(['class', 'sepal_width', 'sepal_length'])

    print(ddf_1.schema())
    print("Number of partitions: ", ddf_1.num_of_partitions())
    print("Number of rows: ", ddf_1.count_rows())


def rename():
    print("\n|-------- With_column Renamed --------|\n")
    data = pd.DataFrame([[i, i + 5, 0] for i in range(10)],
                        columns=['a', 'b', 'c'])
    ddf_1 = DDF().parallelize(data, 4).rename('a', 'A')
    df1 = ddf_1.to_df()
    res_with = pd.DataFrame([[0, 5, 0], [1, 6, 0], [2, 7, 0], [3, 8, 0],
                            [4, 9, 0], [5, 10, 0], [6, 11, 0], [7, 12, 0],
                            [8, 13, 0], [9, 14, 0]], columns=['A', 'b', 'c'])
    assert_frame_equal(df1, res_with, check_index_type=False)
    print("etl_test - with_column - OK")


def range_partition():
    print("\n|-------- Range partition --------|\n")
    n_rows = 1000
    data = pd.DataFrame({'a': np.random.randint(0, 100000, size=n_rows),
                         'b': np.random.randint(0, 100000, size=n_rows),
                         'c': np.random.randint(0, 100000, size=n_rows)
                         })

    ddf_1 = DDF().parallelize(data, 4).range_partition(columns=['a', 'b'],
                                                       nfrag=6)
    f = ddf_1.num_of_partitions()
    print(ddf_1.count_rows(total=False))
    print(f == 6)
    df1 = ddf_1.to_df().sort_values(by=['a', 'b'])
    data = data.sort_values(by=['a', 'b'])
    assert_frame_equal(df1, data, check_index_type=False)
    print("etl_test - repartition - OK")


def replace():
    print("\n|-------- Replace Values --------|\n")
    data = pd.DataFrame([[i, i + 5, 0] for i in range(10)],
                        columns=['a', 'b', 'c'])

    ddf_1 = DDF().parallelize(data, 4).replace({0: 42}, subset=['c'])
    df1 = ddf_1.to_df()
    res_rep = pd.DataFrame([[0, 5, 42], [1, 6, 42], [2, 7, 42], [4, 8, 42],
                            [5, 9, 42], [6, 10, 42], [6, 11, 42], [7, 12, 42],
                            [8, 13, 42], [9, 14, 42]], columns=['a', 'b', 'c'])
    assert_frame_equal(df1, res_rep, check_index_type=False)
    print("etl_test - replace - OK")


def repartition():
    print("\n|-------- Repartition --------|\n")
    data = pd.DataFrame([[i] for i in range(100)],
                        columns=['a'])

    ddf_1 = DDF().parallelize(data, 4).repartition(nfrag=7)
    f = ddf_1.num_of_partitions()
    print(f == 7)
    df1 = ddf_1.to_df()
    assert_frame_equal(df1, data, check_index_type=False)
    print("etl_test - repartition - OK")


def sample():
    print("\n|-------- Sample --------|\n")
    data = pd.DataFrame([[i, i + 5, 0] for i in range(10)],
                        columns=['a', 'b', 'c'])

    ddf_1 = DDF().parallelize(data, 4).sample(7)
    df1 = ddf_1.to_df()
    if len(df1) != 7:
        raise Exception("Sample error")
    print("etl_test - sample - OK")


def save_data_fs():
    print("\n|-------- Save Data in FS--------|\n")
    data = pd.DataFrame([[i, i + 5] for i in range(1000)], columns=['a', 'b'])

    ddf_1 = DDF().parallelize(data, 4)\
        .save('~/test_save_data', storage='fs').to_df('a')
    if len(ddf_1) != n:
        raise Exception("Error in save_data_fs")
    print("etl_test - Save Data - OK")


def save_data_hdfs():
    print("\n|-------- Save Data in HDFS--------|\n")
    n = 10000
    data = pd.DataFrame([[i, i + 5] for i in range(n)], columns=['a', 'b'])

    ddf_1 = DDF().parallelize(data, 4).save('/test_save_data').to_df('a')
    if len(ddf_1) != n:
        raise Exception("Error in save_data_hdfs")
    print("etl_test - Save Data - OK")


def select():
    print("\n|-------- Select --------|\n")
    data = pd.DataFrame([[i, i + 5, 0] for i in range(10)],
                        columns=['a', 'b', 'c'])

    ddf_1 = DDF().parallelize(data, 4).select(['a'])
    df1 = ddf_1.to_df()
    res_rep = pd.DataFrame([[0], [1], [2], [3], [4],  [5], [6], [7],
                            [8], [9]], columns=['a'])
    assert_frame_equal(df1, res_rep, check_index_type=False)
    print("etl_test - select - OK")


def select_expression():
    print("\n|-------- Select Exprs --------|\n")
    data = pd.DataFrame([[i, -i + 5, 1] for i in range(10)],
                        columns=['a', 'b', 'c'])

    ddf_1 = DDF().parallelize(data, 4).select_expression('col2 = a * -1',
                                                         'col3 = col2 * 2 + c',
                                                         'a')
    df1 = ddf_1.to_df()
    res_rep = pd.DataFrame([[0, 1, 0], [-1, -1, 1], [-2, -3, 2], [-3, -5, 3],
                            [-4, -7, 4], [-5, -9, 5], [-6, -11, 6],
                            [-7, -13, 7], [-8, -15, 8], [-9, -17, 9]],
                           columns=['col2', 'col3', 'a'])
    assert_frame_equal(df1, res_rep, check_index_type=False)
    print("etl_test - select exprs - OK")


def show():
    print("\n|-------- Show --------|\n")
    data = pd.DataFrame([[i, -i + 5, 1] for i in range(100)],
                        columns=['a', 'b', 'c'])

    DDF().parallelize(data, 4).show(10)


def sort():
    print("\n|-------- Sort --------|\n")
    power_of2 = [4]  # [2, 4, 8, 16, 32, ]
    not_power = [1, 3, 5, 6, 7, 31, 63]

    for f in power_of2:
        print("# fragments: ", f)
        n1 = np.random.randint(0, 10000, f)

        n1 = sum(n1)
        data = pd.DataFrame({'col0': np.random.randint(1, 1000, n1),
                             'col1': np.random.randint(1, 1000, n1)})
        ddf_1 = DDF().parallelize(data, f)

        # data, info = generate_data(n1, dim=2, max_size=1000)
        # ddf_1 = DDF().import_data(data, info)

        size_b = ddf_1.count_rows(total=False)
        print("size before {}: {}".format(sum(size_b), size_b))
        print("Sorting...")
        t1 = time.time()
        ddf_2 = ddf_1.sort(['col0', 'col1'],
                           ascending=[True, False]).cache()
        t2 = time.time()
        print("... End")
        print('time elapsed: ', t2 - t1)

        size_a = ddf_2.count_rows(total=False)
        print("size after {}: {}".format(sum(size_a), size_a))
        df = ddf_2.to_df()
        a = df['col0'].values

        is_sorted = lambda a: np.all(a[:-1] <= a[1:])
        cond1 = is_sorted(a)
        cond2 = n1 == len(a)
        val = (cond1 and cond2)
        if not val:
            print("error with nfrag=", f)
            print(a)
            print(cond1)
            print(cond2)


def subtract():
    print("\n|-------- Subtract --------|\n")
    cols = ['col1', 'col2']
    s1 = pd.DataFrame([("a", 1), ("a", 1), ("a", 1), ("a", 2), ("b",  3),
                       ("c", 4)], columns=cols)
    s2 = pd.DataFrame([("a", 1), ("b",  3)], columns=cols)

    ddf_1a = DDF().parallelize(s1, 4)
    ddf_1b = DDF().parallelize(s2, 4)
    ddf_2 = ddf_1a.subtract(ddf_1b)
    df1 = ddf_2.to_df()
    print(df1)
    res = pd.DataFrame([("a", 2), ("c",  4)], columns=cols)
    assert_frame_equal(df1, res, check_index_type=False)
    print("etl_test - subtract - OK")


def split():
    print("\n|-------- Split --------|\n")
    size = 100
    data = pd.DataFrame([[i, i+5, 0] for i in range(size)],
                        columns=['a', 'b', 'c'])

    ddf_1a, ddf_1b = DDF().parallelize(data, 4).split(0.5)
    df1 = ddf_1a.to_df()
    df2 = ddf_1b.to_df()

    s = any(pd.concat([df1, df2]).duplicated(['a', 'b', 'c']))
    t = len(df1)+len(df2)
    if s or t != size:
        raise Exception("Split")
    print("etl_test - split - OK")


def take():
    print("\n|-------- Take --------|\n")
    data = pd.DataFrame([[i, i + 5] for i in range(100)], columns=['a', 'b'])
    ddf_1 = DDF().parallelize(data, 4).take(3)

    df1 = ddf_1.to_df()
    res_tak = pd.DataFrame([[0, 5], [1, 6], [2, 7]], columns=['a', 'b'])
    assert_frame_equal(df1, res_tak, check_index_type=False)
    print("etl_test - take - OK")


def union():
    print("\n|-------- Union --------|\n")
    size1 = 20
    size2 = 15
    total_expected = size1 + size2

    data = pd.DataFrame([["left_{}".format(i), 'middle_b']
                         for i in range(size1)], columns=['a', 'b'])
    data1 = pd.DataFrame([["left_{}".format(i), 42, "right_{}".format(i)]
                          for i in range(size1, size1+size2)],
                         columns=['b', 'a', 'c'])

    ddf_1a = DDF().parallelize(data, 4)
    ddf_1b = DDF().parallelize(data1, 4)
    ddf_2 = ddf_1a.union(ddf_1b)
    df1 = ddf_2.to_df()
    print(df1)
    counts = ddf_2.count_rows(total=False)
    print(counts)
    if sum(counts) != total_expected:
        raise Exception('Error in union')


def union_by_name():
    print("\n|-------- Union by Name --------|\n")
    size1 = 3
    size2 = 15
    total_expected = size1 + size2

    data = pd.DataFrame([[i, 5] for i in range(size1)], columns=['a', 'b'])
    data1 = pd.DataFrame([["i{}".format(i), 7, 'c']
                          for i in range(size2)], columns=['b', 'a', 'c'])

    ddf_1a = DDF().parallelize(data, 4)
    ddf_1b = DDF().parallelize(data1, 4)
    ddf_2 = ddf_1a.union_by_name(ddf_1b)
    df1 = ddf_2.to_df()
    print(df1)
    counts = ddf_2.count_rows(total=False)
    print(counts)
    if sum(counts) != total_expected:
        raise Exception('Error in union_by_name')
    print("etl_test - union by name - OK")


if __name__ == '__main__':
    print("_____ETL_____")

    # add_columns()
    # aggregation()
    # balancer()
    # cast()
    # cross_join()
    # etl()
    # except_all()
    # explode()
    # filter_operation()
    # fill_na()  #TODO change dtypes
    # distinct()
    # drop()
    # drop_na()
    # import_data()
    # intersect()
    # intersect_all()
    # join()
    # read_data_single_fs()
    # read_data_multi_fs()
    # read_data_single_hdfs()
    # read_data_multi_hdfs()
    # map()
    # rename()
    # repartition()
    # hash_partition()
    # range_partition()
    # replace()
    # sample()
    # save_data_fs()
    # save_data_hdfs()
    # select()
    # select_expression()
    show()
    # sort()
    # split()
    # subtract()
    # take()
    # union()
    # union_by_name()

