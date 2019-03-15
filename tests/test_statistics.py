#!/usr/bin/python
# -*- coding: utf-8 -*-

from ddf_library.ddf import DDF
import pandas as pd
import numpy as np


def covariance():

    print "\n|-------- Covariance --------|\n"
    # df = pd.DataFrame([[1692, 68], [1978, 102],
    #                    [1884, 110], [2151, 112],
    #                    [2519, 154]], columns=['a', 'b'])
    # cov(a,b) = 9107.3

    df = pd.DataFrame([[1.95, 93.1], [1.96, 93.9], [1.95, 89.9],
                       [1.98, 95.1], [2.10, 100.2]], columns=['a', 'b'])
    # cov(a,b) = 0.2196

    df1 = DDF().parallelize(df, 4).covariance(col1='a', col2='b')
    print df1
    print "etl_test - Covariance - OK"


def correlation():

    print "\n|-------- Correlation --------|\n"

    df = pd.DataFrame([[8, 81], [8, 80], [6, 75],
                       [5, 65], [7, 91], [6, 80]], columns=['a', 'b'])
    # corr(a,b) = 0.6475106

    df1 = DDF().parallelize(df, 4).correlation(col1='a', col2='b')
    print df1
    print "etl_test - Correlation - OK"


def crosstab():
    print "\n|-------- CrossTab --------|\n"
    # data = pd.DataFrame([(1, 1), (1, 2), (2, 1), (2, 1),
    #                      (2, 3), (3, 2), (3, 3)], columns=['key', 'value'])
    data = pd.DataFrame([('v1', 'v1'), ('v1', 'v2'), ('v2', 'v1'), ('v2', 'v1'),
                         ('v2', 'v3'), ('v3', 'v2'), ('v3', 'v3')],
                        columns=['key', 'value'])
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


def describe():
    print "\n|-------- Describe --------|\n"
    data3 = pd.DataFrame([[i, i + 5, 'hello'] for i in xrange(5, 15)],
                         columns=['a', 'b', 'c'])

    data3.loc[15, ['b']] = np.nan
    data3.loc[0:2, ['c']] = 'Hi'
    data3.loc[6:9, ['c']] = 'world'
    data3['d'] = [10, 12, 13, 19, 19, 19, 19, 19, 19, 19, np.nan]

    df1 = DDF().parallelize(data3, 4).describe()
    print df1
    print "etl_test - difference - OK"


def frequent_items():

    print "\n|-------- Frequent Items --------|\n"
    # data = pd.DataFrame([(1, 1), (1, 2), (2, 1), (2, 1),
    #                      (2, 3), (3, 2), (3, 3)], columns=['key', 'value'])
    data = pd.DataFrame([(1, 2, 3) if i % 2 == 0 else (i, 2 * i, i % 4)
                         for i in range(100)], columns=["a", "b", "c"])

    data = pd.DataFrame([(1, -1.0) if (i % 2 == 0) else (i, i * -1.0)
                         for i in range(100)], columns=["a", "b"])
    """
      a_freqItems    b_freqItems
          [1, 99]  [-99.0, -1.0]
    """

    df = DDF().parallelize(data, 4).freq_items(col=['a', 'b'], support=0.4)
    print df


def kolmogorov_smirnov_one_sample():

    print "\n| ------- Kolmogorov Smirnov test -----|\n"

    df = pd.DataFrame()

    # df['a'] = [1.26, 0.34, 0.70, 1.75, 50.57, 1.55, 0.08, 0.42, 0.50, 3.20,
    #            0.15, 0.49, 0.95, 0.24, 1.37, 0.17, 6.98, 0.10, 0.94, 0.38]
    """
    (0.09569124519945327, 0.31907319846415966)
    """

    # df['b'] = [2.37, 2.16, 14.82, 1.73, 41.04, 0.23, 1.32, 2.91, 39.41, 0.11,
    #            27.44, 4.51, 0.51, 4.50, 0.18, 14.68, 4.66, 1.30, 2.06, 1.19]
    """
    (0.6829768039768913, 1.5770173876434368e-08)
    """

    # ks_onesample = DDF().parallelize(df, 4) \
    #     .kolmogorov_smirnov_one_sample(col='b')
    # print ks_onesample

    # gamma
    from scipy.stats import gamma
    np.random.seed(123)
    df['d'] = gamma.rvs(a=15.5, loc=0, scale=1./7, size=100)
    """
    (0.05807585576272023, 0.888689226083446)
    """

    ks_onesample = DDF().parallelize(df, 4)\
        .kolmogorov_smirnov_one_sample(col='d', distribution='gamma',
                                       args=(15.5, 0, 1./7))
    print ks_onesample


if __name__ == '__main__':
    print "_____Statistics_____"
    kolmogorov_smirnov_one_sample()
