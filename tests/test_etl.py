#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from ddf_library.context import COMPSsContext
from ddf_library.utils import generate_info

from pycompss.api.task import task

import pandas as pd
import numpy as np
import time
from pandas.testing import assert_frame_equal


@task(returns=2)
def _generate_partition(size, f, dim, max_size):
    if max_size is None:
        max_size = size * 100

    cols = ["col{}".format(c) for c in range(dim)]
    df = pd.DataFrame({c: np.random.randint(0, max_size, size=size)
                       for c in cols})
    info = generate_info(df, f)
    return df, info


def generate_data(sizes, dim=1, max_size=None):

    nfrag = len(sizes)
    dfs = [[] for _ in range(nfrag)]
    info = [[] for _ in range(nfrag)]

    for f, s in enumerate(sizes):
        dfs[f], info[f] = _generate_partition(s, f, dim, max_size)

    return dfs, info


def check_result(df, size, true, msg, end=False):
    cond = df[:size] == true
    if not cond:
        print(df)
        raise Exception('Error in {}'.format(msg))
    print('{} - ok'.format(msg))
    if end:
        log('{} - FINISHED'.format(msg))


def log(msg):
    print('-' * 50)
    print(msg)


def etl():
    cc = COMPSsContext()
    cc.start_monitor()
    url = ('https://archive.ics.uci.edu/ml/'
           'machine-learning-databases/abalone/abalone.data')
    cols = ['sex', 'length', 'diam', 'height', 'weight', 'rings']
    n_dataset = 20
    data = pd.read_csv(url, usecols=[0, 1, 2, 3, 4, 8], names=cols)[:n_dataset]

    from ddf_library.columns import col, udf
    from ddf_library.types import DataType

    def f1(x):
        return -42 if x > 0.09 else x
    f1_udf = udf(f1, DataType.DECIMAL, col('height'))

    data1 = cc.parallelize(data, 4).map(f1_udf, 'height_nan').cache()

    def f2(x):
        t = '.{}'.format(x)
        return t

    f2_udf = udf(f2, DataType.STRING, col('sex'))
    data2 = data1.map(f2_udf, 'sex')

    log("etl_test_1: Multiple caches")

    df1 = data1.to_df()
    df2 = data2.to_df()
    df3 = data1.to_df()

    check_result(df1['height_nan'].values.tolist(), 5,
                 [-42, 0.09, -42, -42, 0.08], 'etl_test_1a')
    check_result(df2['sex'].values.tolist(), 5, ['.M', '.M', '.F', '.M', '.I'],
                 'etl_test_1b')
    check_result(df3['sex'].values.tolist(), 5, ['M', 'M', 'F', 'M', 'I'],
                 'etl_test_1c', True)

    log("etl_test_2: Branching: data2 and data3 are data1's children. "
        "Note: is this case, those transformations can not be grouped")

    data3 = data1.drop(['length', 'diam']).cache()

    df1 = data1.to_df()
    df2 = data2.to_df()
    df3 = data3.to_df()

    check_result(df1['sex'].values.tolist(), 5, ['M', 'M', 'F', 'M', 'I'],
                 'etl_test_2a')
    check_result(df2['sex'].values.tolist(), 5, ['.M', '.M', '.F', '.M', '.I'],
                 'etl_test_2b')
    check_result(df3['sex'].values.tolist(), 5, ['M', 'M', 'F', 'M', 'I'],
                 'etl_test_2c', True)

    log("etl_test_3: The operations 'drop', 'drop', and 'replace' "
        "must be grouped in a single task")

    data4 = data2.drop(['length']).drop(['diam'])\
        .replace({15: 42}, subset=['rings']).cache()

    df = data4.to_df()
    check_result(df['rings'].values.tolist(), 5,
                 [42, 7, 9, 10, 7], 'etl_test_3')
    check_result(df.columns.tolist(), 5,
                 ['sex', 'height', 'weight', 'rings', 'height_nan'],
                 'etl_test_3', True)

    log("etl_test_4: Check if split (and others operations that returns "
        "more than one output) is working")

    n_total = data4.count_rows()
    data5a, data5b = data4.split(0.40)
    n1 = data5a.count_rows()
    n2 = data5b.count_rows()

    if n1 + n2 != n_total:
        print('data4:', n_total)
        print('data5a:', n1)
        print('data5b:', n2)
        raise Exception('Error in etl_test_4')

    log('etl_test_4 - OK')
    log("etl_test_5: Check if operations with multiple inputs are working")

    data6 = data5b.join(data5a, ['rings'], ['rings'])\
        .filter('(rings > 8)')\
        .select(['sex_l', 'height_l', 'weight_l', 'rings']).cache()

    df = data6.to_df()
    check_result(df.columns.tolist(), 5,
                 ['sex_l', 'height_l', 'weight_l', 'rings'], 'etl_test_5a')

    data7 = data6.sample(7).sort(['rings'], [True])
    data8 = data6.join(data7, ['rings'], ['rings'])

    df = data8.to_df()
    v1 = sum(df[['height_l_l', 'weight_l_l']].values.flatten())
    v2 = sum(df[['height_l_r', 'weight_l_r']].values.flatten())
    cond1 = np.isclose(v1, v2, rtol=0.1)

    cols = data8.schema()['columns'].values.tolist()
    res = ['sex_l_l', 'height_l_l', 'weight_l_l', 'rings',
           'sex_l_r', 'height_l_r', 'weight_l_r']
    cond2 = cols == res
    if not (cond1 and cond2):
        raise Exception('Error in etl_test_5b')
    log('etl_test_5b - OK')
    log("etl_test_76: Check if 'count_rows' and 'take' are working.")
    n = 7
    v = data1.select(['rings']).count_rows()
    len_df = data1.select(['rings']).take(n).count_rows()

    cond = v != n_dataset
    if cond:
        print(v)
        raise Exception('Error in etl_test_6a')

    cond = len_df != n
    if cond:
        print(len_df)
        raise Exception('Error in etl_test_6b')
    log("etl_test_7b - OK")

    import time
    time.sleep(5)
    cc.context_status()
    # cc.show_tasks()
    cc.stop()


def add_columns():
    print("\n|-------- Add Column --------|\n")
    data1 = pd.DataFrame([["N_{}".format(i)] for i in range(5)],
                         columns=['name'])

    data2 = pd.DataFrame([["A_{}".format(i), i + 5] for i in range(5, 15)],
                         columns=['name', 'b'])
    cc = COMPSsContext()
    ddf_1a = cc.parallelize(data1, 5)
    ddf_1b = cc.parallelize(data2, 10)
    df1 = ddf_1a.add_column(ddf_1b).to_df()

    res = pd.merge(data1, data2, left_index=True,  right_index=True,
                   suffixes=['_l', '_r'])

    assert_frame_equal(df1, res, check_index_type=False)

    print("etl_test - add column - OK")
    cc.stop()


def aggregation():
    print("\n|-------- Aggregation --------|\n")
    n = 10
    data3 = pd.DataFrame([[i, i + 5, 'hello'] for i in range(n)],
                         columns=['a', 'b', 'c'])
    cc = COMPSsContext()
    ddf_1 = cc.parallelize(data3, 4).group_by(['c']).agg(
            count_a=('a', 'count'),
            first_b=('b', 'first'),
            last_b=('b', 'last')
    )
    df = ddf_1.to_df()
    print (df)
    cond1 = len(df) == 1
    cond2 = all([f == n for f in df['count_a'].values])
    if not (cond1 and cond2):
        print(df)
        raise Exception('Error in aggregation')

    ddf_2 = cc.parallelize(data3, 4).group_by(['a', 'c']).count('*')

    df = ddf_2.to_df()
    cond1 = len(df) == n
    cond2 = all([f == 1 for f in df['count(*)'].values])
    if not (cond1 and cond2):
        print(df)
        raise Exception('Error in aggregation')

    ddf3 = cc.parallelize(data3, 4).group_by('c').list('*')
    print(ddf3.to_df())

    ddf3 = cc.parallelize(data3, 4).group_by(['c']).set('*')
    print(ddf3.to_df())

    print("etl_test - aggregation - OK")
    cc.stop()


def balancer():
    print("\n|-------- Balance --------|\n")

    iterations = [[10, 0, 10, 5, 100],
                  # [100, 5, 10, 0, 10],
                  # [85, 0, 32, 0, 0],
                  # [0, 0, 0, 30, 100]
                  ]
    cc = COMPSsContext()
    cc.set_log(True)
    for s in iterations:
        print('Before:', s)
        data, info = generate_data(s)
        ddf_1 = cc.import_compss_data(data, schema=info, parquet=False).cache()
        df1 = ddf_1.to_df()['col0'].values

        ddf_2 = ddf_1.balancer(forced=True) #.cache()
        size_a = ddf_2.count_rows(total=False)
        df2 = ddf_2.to_df()['col0'].values

        print('After:', size_a)
        print(np.array_equal(df1, df2))
    cc.stop()


def cast():
    print("\n|-------- cast --------|\n")
    data = pd.DataFrame([[i, i + 5, 0] for i in range(10)],
                        columns=['a', 'b', 'c'])
    cc = COMPSsContext()
    ddf_1 = cc.parallelize(data, 4).cast(['a', 'b'], 'string')
    schema = ddf_1.schema()
    print(schema)
    print("etl_test - cast - OK")
    cc.stop()


def cross_join():
    print("\n|-------- CrossJoin --------|\n")
    data1 = pd.DataFrame([["Bob_{}".format(i), i + 5] for i in range(5)],
                         columns=['name', 'height'])
    data2 = pd.DataFrame([[i + 5] for i in range(5, 15)], columns=['gain'])
    cc = COMPSsContext()
    ddf_1a = cc.parallelize(data1, 4)
    ddf_1b = cc.parallelize(data2, 4)
    df1 = ddf_1a.cross_join(ddf_1b).to_df().sort_values(by=['name', 'gain'])
    print(df1[0:50])
    cc.stop()


def distinct():
    print("\n|-------- Distinct --------|\n")
    data = pd.DataFrame([[i, i + 5, 0] for i in range(10)],
                        columns=['a', 'b', 'c'])
    cc = COMPSsContext()
    ddf_1 = cc.parallelize(data, 4).distinct(['c'], opt=True)
    df1 = ddf_1.cache().to_df()
    print(df1)
    res_dist = pd.DataFrame([[0, 5, 0]], columns=['a', 'b', 'c'])
    assert_frame_equal(df1, res_dist, check_index_type=False)
    print("etl_test - distinct - OK")
    cc.stop()


def drop():
    print("\n|-------- Drop --------|\n")
    data = pd.DataFrame([[i, i + 5, 0] for i in range(10)],
                        columns=['a', 'b', 'c'])
    cc = COMPSsContext()
    ddf_1 = cc.parallelize(data, 4).drop(['a'])
    df1 = ddf_1.to_df()
    res_drop = pd.DataFrame([[5, 0], [6, 0], [7, 0], [8, 0], [9, 0],
                             [10, 0], [11, 0], [12, 0],
                             [13, 0], [14, 0]], columns=['b', 'c'])
    assert_frame_equal(df1, res_drop, check_index_type=False)
    print("etl_test - drop - OK")
    cc.stop()


def drop_na():
    print("\n|-------- DropNaN --------|\n")
    data3 = pd.DataFrame([[i, i + 5, 'hello'] for i in range(5, 15)],
                         columns=['a', 'b', 'c'])

    data3.loc[15, ['b']] = np.nan
    data3['d'] = [10, 12, 13, 19, 19, 19, 19, 19, 19, 19, np.nan]
    data3['g'] = [10, 12, 13, 19, 19, 19, 19, 19, 19, np.nan, np.nan]
    cc = COMPSsContext()
    ddf_1 = cc.parallelize(data3, 4)
    df1a = ddf_1.dropna(['c'], mode='REMOVE_COLUMN', how='all', thresh=1)
    df1b = ddf_1.dropna(['c'], mode='REMOVE_ROW', how='any')

    print(df1a.to_df())
    print(df1b.to_df())
    cc.stop()


def except_all():
    print("\n|-------- ExceptAll --------|\n")
    cols = ['a', 'b']
    s1 = pd.DataFrame([("a", 1), ("a", 1), ("a", 1), ("a", 2), ("b",  3),
                       ("c", 4)], columns=cols)
    s2 = pd.DataFrame([("a", 1), ("b",  3), ('e', 4), ('e', 4), ('e', 4),
                       ('e', 6), ('e', 9), ('e', 10), ('e', 4), ('e', 4)],
                      columns=cols)
    cc = COMPSsContext()
    ddf_1a = cc.parallelize(s1, 2)
    ddf_1b = cc.parallelize(s2, 4)
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
    cc.stop()


def explode():
    print("\n|-------- Explode --------|\n")

    df_size = 1 * 10 ** 3

    df = pd.DataFrame(np.random.randint(1, df_size, (df_size, 2)),
                      columns=list("AB"))
    df['C'] = df[['A', 'B']].values.tolist()

    col = 'C'
    cc = COMPSsContext()
    ddf1 = cc.parallelize(df, 4).explode(col)
    ddf1.show()
    print("etl_test - explode - OK")
    cc.stop()


def filter_operation():
    print("\n|-------- Filter --------|\n")
    data = pd.DataFrame([[i, i + 5] for i in range(10)], columns=['a', 'b'])
    cc = COMPSsContext()
    ddf_1 = cc.parallelize(data, 4).filter('a > 5')
    df1 = ddf_1.to_df()
    res_fil = pd.DataFrame([[6, 11], [7, 12],
                            [8, 13], [9, 14]], columns=['a', 'b'])
    assert_frame_equal(df1, res_fil, check_index_type=False)
    print("etl_test - filter - OK")
    cc.stop()


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
    cc = COMPSsContext()
    ddf_1 = cc.parallelize(data3, 4)
    # df1a = ddf_1.fillna(mode='VALUE', value=42),
    # df1a = ddf_1.fillna(mode='VALUE', value={'c': 42})
    # df1a = ddf_1.fillna(['a', 'b'], mode='MEAN')
    # df1a = ddf_1.fillna(['c'], mode='MODE')

    df1a = ddf_1.fillna(['a', 'b', 'd','e','f','g', 'h', 'i'], mode='MEDIAN')

    print(df1a.to_df())
    print("A: 9.5 - B: 14.5 - D: 19.0 - E: 10.0 - G: 19.0 - H: 5.0 - I: 8.5")
    cc.stop()


def flow_serial_only():
    print("\n|-------- Flow to test serial tasks --------|\n")

    data = pd.DataFrame([[i, i + 5, 'hello', i + 7] for i in range(1, 15)],
                        columns=['a', 'b', 'c', 'd'])

    from ddf_library.types import DataType
    from ddf_library.columns import col, udf

    def f3(x):
        return 7 if x > 5 else x

    cat = udf(f3, DataType.INT, col('a'))

    cc = COMPSsContext()
    ddf1 = cc.parallelize(data, '*') \
        .map(cat, 'e')\
        .drop(['c'])\
        .select(['a', 'b', 'd'])\
        .select(['a', 'b']).to_df()

    print(ddf1)
    cc.show_tasks()
    cc.stop()


def flow_recompute_task():
    print("\n|-------- Flow to test task re-computation --------|\n")
    cc = COMPSsContext()
    # cc.set_log(True)
    data = pd.DataFrame([[i, i + 5, 'hello', i + 7] for i in range(1, 25)],
                        columns=['a', 'b', 'c', 'd'])
    ddf1 = cc.parallelize(data, '*')\
        .drop(['c']) \
        .sample(10)

    ddf2 = ddf1.distinct(['a'])\
        .select(['a', 'b', 'd'])\
        .select(['a', 'b']) \
        .select(['a'])\
        .sample(5).select(['a'])
    ddf2.save.csv('file:///tmp/flow_recompute_task')

    ddf3 = ddf1.select(['a', 'b'])
    ddf3.show()
    cc.context_status()
    cc.stop()
    # expected result: 1 temporary output (select) and 1 persisted (save)


def hash_partition():
    print("\n|-------- Hash partition --------|\n")
    n_rows = 1000
    data = pd.DataFrame({'a': np.random.randint(0, 100000, size=n_rows),
                         'b': np.random.randint(0, 100000, size=n_rows),
                         'c': np.random.randint(0, 100000, size=n_rows)
                         })
    data['b'] = data['b'].astype(str)
    cc = COMPSsContext()
    ddf_1 = cc.parallelize(data, 12).hash_partition(columns=['a', 'b'],
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
    cc.stop()


def import_data():
    print("\n|-------- Import data --------|\n")
    s1 = pd.DataFrame([("a", 1), ("a", 1), ("a", 1), ("a", 2), ("b", 3),
                       ("c", 4)], columns=['col1', 'col2'])
    cc = COMPSsContext()
    df1 = cc.import_compss_data(np.array_split(s1, 4)).to_df()
    assert_frame_equal(df1, s1, check_index_type=False)
    print("etl_test - import data - OK")
    cc.stop()


def intersect():
    print("\n|-------- Intersect --------|\n")
    cols = ['col1', 'col2']
    s1 = pd.DataFrame([("a", 1), ("a", 1), ("a", 1), ("a", 2), ("b", 3),
                       ("c", 4)], columns=cols)
    s2 = pd.DataFrame([('a', 1), ('a', 1), ('b', 3)], columns=cols)
    cc = COMPSsContext()
    ddf_1a = cc.parallelize(s1, 4)
    ddf_1b = cc.parallelize(s2, 4)
    ddf_2 = ddf_1a.intersect(ddf_1b)

    df1 = ddf_2.to_df().sort_values(by=cols)
    res = pd.DataFrame([['b', 3], ['a', 1]], columns=cols)
    res.sort_values(by=cols, inplace=True)

    assert_frame_equal(df1, res, check_index_type=False)
    print("etl_test - intersect - OK")
    cc.stop()


def intersect_all():
    print("\n|-------- Intersect All--------|\n")
    cols = ['col1', 'col2']
    s1 = pd.DataFrame([('a', 1), ('a', 1), ('b', 3), ('c', 4)], columns=cols)
    s2 = pd.DataFrame([('a', 1), ('a', 1), ('b', 3)], columns=cols)
    cc = COMPSsContext()
    ddf_1a = cc.parallelize(s1, 4)
    ddf_1b = cc.parallelize(s2, 3)
    ddf_2 = ddf_1a.intersect_all(ddf_1b)

    df1 = ddf_2.to_df().sort_values(by=cols)
    res = pd.DataFrame([['b', 3], ['a', 1], ['a', 1]], columns=cols)
    res.sort_values(by=cols, inplace=True)

    assert_frame_equal(df1, res, check_index_type=False)
    print("etl_test - intersect all - OK")
    cc.stop()


def join():
    print("\n|--------  inner join --------|\n")

    data1 = pd.DataFrame([[i, i + 5, 0] for i in range(10)],
                         columns=['a', 'b', 'c'])
    data2 = data1.copy()
    data2.sample(frac=1,  replace=False)
    cc = COMPSsContext()
    ddf_1a = cc.parallelize(data1, 5)
    ddf_1b = cc.parallelize(data2, 3)
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
    cc = COMPSsContext()
    ddf_1a = cc.parallelize(data1, 4)
    ddf_1b = cc.parallelize(data2, 4)
    ddf_2 = ddf_1a.join(ddf_1b, key1=['a'], key2=['a'], mode='left')
    df1 = ddf_2.to_df().sort_values(by=['a'])
    print(df1)

    print("etl_test - left join - OK")

    print("\n|--------  right join --------|\n")
    data1 = pd.DataFrame([[i, i + 5, 0] for i in range(100)],
                         columns=['a', 'b', 'c'])
    data1['b'] = data1['b'].astype('int8')
    data1['c'] = data1['c'].astype('int8')

    data2 = data1.copy()
    data1 = data1[0:50]
    data2.sample(frac=1,  replace=False)
    data2.drop(['b', 'c'], axis=1, inplace=True)
    data2['d'] = 'd'
    cc = COMPSsContext()
    ddf_1a = cc.parallelize(data1, 4)
    ddf_1b = cc.parallelize(data2, 4)
    ddf_2 = ddf_1a.join(ddf_1b, key1=['a'], key2=['a'], mode='right')
    df1 = ddf_2.to_df().sort_values(by=['a'])
    print(df1)

    print("etl_test - right join - OK")
    cc.stop()


def map():
    print("\n|-------- Map operation --------|\n")
    data = pd.DataFrame([[i, i + 5, 0] for i in range(10)],
                        columns=['a', 'b', 'c'])
    data['date'] = ['30/04/17 1{}:46:5{}.000000'.format(i, i)
                    for i in range(10)]

    from ddf_library.types import DataType
    from ddf_library.columns import col, udf

    def f3(x):
        return 7 if x > 5 else x

    cat = udf(f3, DataType.INT, col('a'))
    cc = COMPSsContext()
    ddf_1 = cc.parallelize(data, 4) \
        .map(col('b').cast(DataType.DECIMAL), 'd')\
        .map(cat, 'a')\
        .map(col('date').to_datetime('dd/MM/yy HH:mm:ss.SSSSSS'), 'e')\
        .map(col('e').year(), 'f')

    df1 = ddf_1.to_df()
    print(df1.dtypes)
    print(df1)
    res_tra = pd.DataFrame([[0, 5, 0], [1, 6, 0], [2, 7, 0], [3, 8, 0],
                            [4, 9, 0], [5, 10, 0], [7, 11, 0], [7, 12, 0],
                            [7, 13, 0], [7, 14, 0]], columns=['a', 'b', 'c'])
    assert_frame_equal(df1, res_tra, check_index_type=False)
    print("etl_test - map - OK")
    cc.stop()


def read_data_single_fs():
    print("\n|-------- Read Data from a single file on FS --------|\n")
    from ddf_library.types import DataType
    cc = COMPSsContext()
    dtypes = {'sepal_length': DataType.DECIMAL, 'sepal_width': DataType.DECIMAL,
              'petal_length': DataType.DECIMAL, 'petal_width': DataType.DECIMAL,
              'class': DataType.STRING}
    ddf_1 = cc.read.csv('file://./datasets/iris-dataset.csv', header=True,
                        sep=',', schema=dtypes)\
        .select(['class', 'sepal_length'])\
        #.save.csv('file:///tmp/read_data_single_fs')

    print(ddf_1.schema())
    print("Number of partitions: ", ddf_1.num_of_partitions())
    print("Number of rows: ", ddf_1.count_rows())
    cc.stop()


def read_data_multi_fs():
    print("\n|-------- Read Data from files in a folder on FS --------|\n")
    from ddf_library.types import DataType
    cc = COMPSsContext()
    dtypes = {'sepal_length': DataType.DECIMAL, 'sepal_width': DataType.DECIMAL,
              'petal_length': DataType.DECIMAL, 'petal_width': DataType.DECIMAL,
              'class': DataType.STRING}
    ddf_1 = cc.read.csv('file://./datasets/iris_dataset_folder/', header=True,
                        sep=',', schema=dtypes)\
        .select(['class', 'sepal_width'])\
        # .save.csv('file:///tmp/read_data_multi_fs')

    print(ddf_1.schema())
    print("Number of partitions: ", ddf_1.num_of_partitions())
    print("Number of rows: ", ddf_1.count_rows())
    cc.stop()


def read_data_single_hdfs():
    print("\n|-------- Read Data From a single file on HDFS --------|\n")
    from ddf_library.types import DataType
    cc = COMPSsContext()
    dtypes = {'sepal_length': DataType.DECIMAL, 'sepal_width': DataType.DECIMAL,
              'petal_length': DataType.DECIMAL, 'petal_width': DataType.DECIMAL,
              'class': DataType.STRING}
    ddf_1 = cc.read.csv('hdfs://localhost:9000/iris-dataset.csv',
                           header=True, sep=',', schema=dtypes)\
        .select(['sepal_length'])

    print(ddf_1.schema())
    print(ddf_1.count_rows())
    print("Number of partitions: ", ddf_1.num_of_partitions())
    print("Number of rows: ", ddf_1.count_rows())
    cc.stop()


def read_data_multi_hdfs():
    print("\n|-------- Read Data from files in a folder on HDFS --------|\n")
    from ddf_library.types import DataType
    cc = COMPSsContext()
    dtypes = {'sepal_length': DataType.DECIMAL, 'sepal_width': DataType.DECIMAL,
              'petal_length': DataType.DECIMAL, 'petal_width': DataType.DECIMAL,
              'class': DataType.STRING}
    ddf_1 = cc.read.csv('hdfs://localhost:9000/iris_dataset_folder/',
                           header=True, sep=',', schema=dtypes)\
        .select(['class', 'sepal_width', 'sepal_length'])

    print(ddf_1.schema())
    print("Number of partitions: ", ddf_1.num_of_partitions())
    print("Number of rows: ", ddf_1.count_rows())
    cc.stop()


def rename():
    print("\n|-------- With_column Renamed --------|\n")
    data = pd.DataFrame([[i, i + 5, 0] for i in range(10)],
                        columns=['a', 'b', 'c'])
    cc = COMPSsContext()
    ddf_1 = cc.parallelize(data, 4).rename('a', 'A')
    df1 = ddf_1.to_df()
    res_with = pd.DataFrame([[0, 5, 0], [1, 6, 0], [2, 7, 0], [3, 8, 0],
                            [4, 9, 0], [5, 10, 0], [6, 11, 0], [7, 12, 0],
                            [8, 13, 0], [9, 14, 0]], columns=['A', 'b', 'c'])
    assert_frame_equal(df1, res_with, check_index_type=False)
    print("etl_test - with_column - OK")
    cc.stop()


def range_partition():
    print("\n|-------- Range partition --------|\n")
    n_rows = 1000
    data = pd.DataFrame({'a': np.random.randint(0, 100000, size=n_rows),
                         'b': np.random.randint(0, 100000, size=n_rows),
                         'c': np.random.randint(0, 100000, size=n_rows)
                         })
    cc = COMPSsContext()
    ddf_1 = cc.parallelize(data, 4).range_partition(columns=['a', 'b'],
                                                       nfrag=6)
    f = ddf_1.num_of_partitions()
    print(ddf_1.count_rows(total=False))
    print(f == 6)
    df1 = ddf_1.to_df().sort_values(by=['a', 'b'])
    data = data.sort_values(by=['a', 'b'])
    assert_frame_equal(df1, data, check_index_type=False)
    print("etl_test - repartition - OK")
    cc.stop()


def replace():
    print("\n|-------- Replace Values --------|\n")
    data = pd.DataFrame([[i, i + 5, 0] for i in range(10)],
                        columns=['a', 'b', 'c'])
    cc = COMPSsContext()
    ddf_1 = cc.parallelize(data, 4).replace({0: 42}, subset=['c'])
    df1 = ddf_1.to_df()
    res_rep = pd.DataFrame([[0, 5, 42], [1, 6, 42], [2, 7, 42], [4, 8, 42],
                            [5, 9, 42], [6, 10, 42], [6, 11, 42], [7, 12, 42],
                            [8, 13, 42], [9, 14, 42]], columns=['a', 'b', 'c'])
    assert_frame_equal(df1, res_rep, check_index_type=False)
    print("etl_test - replace - OK")
    cc.stop()


def repartition():
    print("\n|-------- Repartition --------|\n")
    data = pd.DataFrame([[i] for i in range(100)],
                        columns=['a'])
    cc = COMPSsContext()
    ddf_1 = cc.parallelize(data, 4).repartition(nfrag=7)
    f = ddf_1.num_of_partitions()
    print(f == 7)
    df1 = ddf_1.to_df()
    assert_frame_equal(df1, data, check_index_type=False)
    print("etl_test - repartition - OK")
    cc.stop()


def sample():
    print("\n|-------- Sample --------|\n")
    data = pd.DataFrame([[i, i + 5, 0] for i in range(10)],
                        columns=['a', 'b', 'c'])
    cc = COMPSsContext()
    ddf_1 = cc.parallelize(data, 4).sample(7)
    df1 = ddf_1.to_df()
    if len(df1) != 7:
        raise Exception("Sample error")
    print("etl_test - sample - OK")
    cc.stop()


def save_data_fs():
    print("\n|-------- Save Data in FS--------|\n")
    n = 1000
    data = pd.DataFrame([[i, i + 5] for i in range(n)], columns=['a', 'b'])
    cc = COMPSsContext()
    path = 'file:///tmp/test_save_data'
    ddf_1 = cc.parallelize(data, 4)\
        .save.csv(path)

    import os
    if not os.path.isdir(path.replace('file://', '')):
        raise Exception("Error in save_data_fs")
    log("etl_test - Save Data - OK")
    cc.stop()


def save_data_hdfs():
    print("\n|-------- Save Data in HDFS--------|\n")
    n = 10000
    data = pd.DataFrame([[i, i + 5] for i in range(n)], columns=['a', 'b'])

    path = 'hdfs://localhost:9000/test_save_data'

    from hdfspycompss.hdfs import HDFS
    dfs = HDFS(host='localhost', port=9000)
    if dfs.exist('/test_save_data'):
        dfs.rm('/test_save_data', recursive=True)
    cc = COMPSsContext()
    ddf_1 = cc.parallelize(data, 4)\
        .select(['a', 'b'])\
        .save.csv(path)

    if not dfs.exist(path):
        raise Exception("Error in save_data_hdfs")
    log("etl_test - Save Data - OK")
    cc.stop()


def select():
    print("\n|-------- Select --------|\n")
    data = pd.DataFrame([[i, i + 5, 0] for i in range(10)],
                        columns=['a', 'b', 'c'])
    cc = COMPSsContext()
    ddf_1 = cc.parallelize(data, 4).select(['a'])
    df1 = ddf_1.to_df()
    res_rep = pd.DataFrame([[0], [1], [2], [3], [4],  [5], [6], [7],
                            [8], [9]], columns=['a'])
    assert_frame_equal(df1, res_rep, check_index_type=False)
    print("etl_test - select - OK")
    cc.stop()


def select_expression():
    print("\n|-------- Select Exprs --------|\n")
    data = pd.DataFrame([[i, -i + 5, 1] for i in range(10)],
                        columns=['a', 'b', 'c'])
    cc = COMPSsContext()
    ddf_1 = cc.parallelize(data, 4).select_expression('col2 = a * -1',
                                                      'col3 = col2 * 2 + c',
                                                      'a')
    df1 = ddf_1.to_df()
    res_rep = pd.DataFrame([[0, 1, 0], [-1, -1, 1], [-2, -3, 2], [-3, -5, 3],
                            [-4, -7, 4], [-5, -9, 5], [-6, -11, 6],
                            [-7, -13, 7], [-8, -15, 8], [-9, -17, 9]],
                           columns=['col2', 'col3', 'a'])
    assert_frame_equal(df1, res_rep, check_index_type=False)
    print("etl_test - select exprs - OK")
    cc.stop()


def show():
    print("\n|-------- Show --------|\n")
    data = pd.DataFrame([[i, -i + 5, 1] for i in range(100)],
                        columns=['a', 'b', 'c'])
    cc = COMPSsContext()
    cc.parallelize(data, 4).show(10)
    cc.stop()


def sort():
    print("\n|-------- Sort --------|\n")
    power_of2 = [4]  # [2, 4, 8, 16, 32, ]
    not_power = [1, 3, 5, 6, 7, 31, 63]
    cc = COMPSsContext()
    for f in power_of2:
        print("# fragments: ", f)
        n1 = np.random.randint(0, 10000, f)

        n1 = sum(n1)
        data = pd.DataFrame({'col0': np.random.randint(1, 1000, n1),
                             'col1': np.random.randint(1, 1000, n1)})
        ddf_1 = cc.parallelize(data, f)

        # data, schema = generate_data(n1, dim=2, max_size=1000)
        # ddf_1 = cc.import_compss_data(data, schema)

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
    cc.stop()


def subtract():
    print("\n|-------- Subtract --------|\n")
    cols = ['col1', 'col2']
    s1 = pd.DataFrame([("a", 1), ("a", 1), ("a", 1), ("a", 2), ("b",  3),
                       ("c", 4)], columns=cols)
    s2 = pd.DataFrame([("a", 1), ("b",  3)], columns=cols)
    cc = COMPSsContext()
    ddf_1a = cc.parallelize(s1, 4)
    ddf_1b = cc.parallelize(s2, 2)
    ddf_2 = ddf_1a.subtract(ddf_1b)
    df1 = ddf_2.to_df()

    res = pd.DataFrame([("a", 2), ("c",  4)], columns=cols)
    assert_frame_equal(df1, res, check_index_type=False)
    print("etl_test - subtract - OK")
    cc.stop()


def split():
    print("\n|-------- Split --------|\n")
    size = 100
    data = pd.DataFrame([[i, i+5, 0] for i in range(size)],
                        columns=['a', 'b', 'c'])
    cc = COMPSsContext()
    ddf_1a, ddf_1b = cc.parallelize(data, 4).split(0.25)
    df1 = ddf_1a.to_df()
    df2 = ddf_1b.to_df()

    s = any(pd.concat([df1, df2]).duplicated(['a', 'b', 'c']))
    t = len(df1)+len(df2)

    if s or t != size:
        raise Exception("Split")
    print("etl_test - split - OK")
    cc.stop()


def take():
    print("\n|-------- Take --------|\n")
    data = pd.DataFrame([[i, i + 5] for i in range(100)], columns=['a', 'b'])
    cc = COMPSsContext()
    ddf_1 = cc.parallelize(data, 4).take(40)

    dfs = ddf_1.to_df(split=True)
    for df in dfs:
        if len(df) != 10:
            raise Exception("Error in take()")
    log("etl_test - take - OK")
    cc.stop()


def union():
    print("\n|-------- Union --------|\n")
    size1 = 20
    size2 = 15
    total_expected = size1 + size2
    cc = COMPSsContext()
    data = pd.DataFrame([["left_{}".format(i), 'middle_b']
                         for i in range(size1)], columns=['a', 'b'])
    data1 = pd.DataFrame([[42, "right_{}".format(i)]
                          for i in range(size1, size1+size2)],
                         columns=['b', 'c'])

    ddf_1a = cc.parallelize(data, 4)
    ddf_1b = cc.parallelize(data1, 4)
    ddf_2 = ddf_1a.union(ddf_1b)
    df1 = ddf_2.to_df()
    print(df1)
    counts = ddf_2.count_rows(total=False)
    print(counts)
    if sum(counts) != total_expected:
        raise Exception('Error in union')
    cc.stop()


def union_by_name():
    print("\n|-------- Union by Name --------|\n")
    size1 = 3
    size2 = 15
    total_expected = size1 + size2

    data = pd.DataFrame([[i, 5] for i in range(size1)], columns=['a', 'b'])
    data1 = pd.DataFrame([["i{}".format(i), 7, 'c']
                          for i in range(size2)], columns=['b', 'a', 'c'])
    cc = COMPSsContext()
    ddf_1a = cc.parallelize(data, 4)
    ddf_1b = cc.parallelize(data1, 4)
    ddf_2 = ddf_1a.union_by_name(ddf_1b)
    df1 = ddf_2.to_df()
    print(df1)
    counts = ddf_2.count_rows(total=False)
    print(counts)
    if sum(counts) != total_expected:
        raise Exception('Error in union_by_name')
    print("etl_test - union by name - OK")
    cc.stop()


if __name__ == '__main__':
    print("_____ETL_____")

    import argparse

    parser = argparse.ArgumentParser(description="ETL operations")
    parser.add_argument('-o', '--operation',
                        type=str,
                        required=True,
                        help="""
            add_columns, aggregation, balancer, cast, 
            cross_join, etl, except_all, explode, filter, fill_na,
            flow_serial_only, flow_recompute_task, distinct, drop, drop_na,
            import_compss_data, intersect, intersect_all, join, read_data_single_fs,
            read_data_multi_fs, read_data_single_hdfs, read_data_multi_hdfs,
            map, rename, repartition, hash_partition, range_partition,
            replace, sample, save_data_fs, save_data_hdfs, select,
            select_expression, show, sort, split, subtract, take, 
            union, union_by_name""")
    arg = vars(parser.parse_args())

    operation = arg['operation']
    operations = dict()
    operations['add_columns'] = add_columns
    operations['aggregation'] = aggregation
    operations['balancer'] = balancer
    operations['cast'] = cast
    operations['cross_join'] = cross_join
    operations['etl'] = etl
    operations['except_all'] = except_all
    operations['explode'] = explode
    operations['filter'] = filter_operation
    operations['fill_na'] = fill_na
    operations['flow_serial_only'] = flow_serial_only
    operations['flow_recompute_task'] = flow_recompute_task
    operations['distinct'] = distinct
    operations['drop'] = drop
    operations['drop_na'] = drop_na
    operations['hash_partition'] = hash_partition
    operations['import_compss_data'] = import_data
    operations['intersect'] = intersect
    operations['intersect_all'] = intersect_all
    operations['join'] = join
    operations['read_data_single_fs'] = read_data_single_fs
    operations['read_data_multi_fs'] = read_data_multi_fs
    operations['read_data_single_hdfs'] = read_data_single_hdfs
    operations['read_data_multi_hdfs'] = read_data_multi_hdfs
    operations['map'] = map
    operations['rename'] = rename
    operations['range_partition'] = range_partition
    operations['repartition'] = repartition
    operations['replace'] = replace
    operations['sample'] = sample
    operations['save_data_fs'] = save_data_fs
    operations['save_data_hdfs'] = save_data_hdfs
    operations['select'] = select
    operations['select_expression'] = select_expression
    operations['show'] = show
    operations['sort'] = sort
    operations['split'] = split
    operations['subtract'] = subtract
    operations['take'] = take
    operations['union'] = union
    operations['union_by_name'] = union_by_name

    operations[operation]()

