#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pycompss.api.task import task
from pycompss.api.api import compss_barrier

from ddf_library.ddf import DDF
from ddf_library.utils import generate_info
from ddf_library.functions.ml.feature import StringIndexer, StandardScaler

import pandas as pd
import numpy as np
import time
import sys
import math
from io import StringIO

####
#   Data generation
####

dtypes = {'PassengerId': np.int16, 'Survived': np.int8, 'Pclass': np.int8,
          'Name': np.dtype('O'), 'Sex': np.dtype('O'), 'Age': np.float16,
          'SibSp': np.int8, 'Parch': np.int8, 'Ticket': np.dtype('O'),
          'Fare': np.float16, 'Cabin': np.dtype('O'),
          'Embarked': np.dtype('O')}

# @task(returns=2)
# def generate_partition(x, multiplier, frag):
#     x = x * multiplier
#     cols = ['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age',
#             'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']
#     df = pd.read_csv(StringIO(x), sep=',',
#                      names=cols, header=None, dtype=dtypes)
#     schema = generate_info(df, frag)
#     return df, schema
#
#
# def generate_data(n_repeat, nfrag):
#
#     dfs = [[] for _ in range(nfrag)]
#     info = [[] for _ in range(nfrag)]
#
#     block = ''
#     with open('base_titanic.csv', 'r') as content_file:
#         block = content_file.read()
#
#     for f in range(nfrag):
#         dfs[f], info[f] = generate_partition(block, n_repeat, f)
#
#     return dfs, info


if __name__ == "__main__":

    ####
    #   Data generation
    ####
    print("Titanic workflow implemented to DDF v0.4")
    n_frag = int(sys.argv[2])

    t1 = time.time()
    # if working with artificial data artificial data
    # size_block = int(sys.argv[1])*1024*1024  # in bytes
    # size_original = 61113
    # n_repeat = int(math.ceil(size_block / size_original))
    # df_list, info = generate_data(n_repeat, n_frag)
    # ddf1 = DDF().import_data(df_list, info)
    ddf1 = DDF().load_text('hdfs://localhost:9000/titanic.csv',
                           num_of_parts=n_frag,
                           dtypes=dtypes)

    print("Number of rows: ", ddf1.count_rows())
    compss_barrier()
    t2 = time.time()
    print("Time to generate and import data - t2-t1:", t2 - t1)

    def title_checker(row):
        from ddf_library.utils import col
        titles = {"Mr.": 1, "Miss": 2, "Mrs.": 3, "Master": 4, "Rare": 5}
        for title in titles:
            if title in row[col('Name')]:
                return titles[title]
        return -1

    def age_categorizer(row):
        from ddf_library.utils import col
        category = 7

        if row[col('Age')] <= 11:
            category = 0
        elif (row[col('Age')] > 11) and (row[col('Age')] <= 18):
            category = 1
        elif (row[col('Age')] > 18) and (row[col('Age')] <= 22):
            category = 2
        elif (row[col('Age')] > 22) and (row[col('Age')] <= 27):
            category = 3
        elif (row[col('Age')] > 27) and (row[col('Age')] <= 33):
            category = 4
        elif (row[col('Age')] > 33) and (row[col('Age')] <= 40):
            category = 5
        elif (row[col('Age')] > 40) and (row[col('Age')] <= 66):
            category = 6

        return category

    def fare_categorizer(row):
        from ddf_library.utils import col
        category = 5
        if row[col('Fare')] <= 7.91:
            category = 0
        elif (row[col('Fare')] > 7.91) and (row[col('Fare')] <= 14.454):
            category = 1
        elif (row[col('Fare')] > 14.454) and (row[col('Fare')] <= 31):
            category = 2
        elif (row[col('Fare')] > 31) and (row[col('Fare')] <= 99):
            category = 3
        elif (row[col('Fare')] > 99) and (row[col('Fare')] <= 250):
            category = 4
        return category


    features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
    all_columns = features + ['Survived']

    ddf1 = StringIndexer(input_col='Embarked') \
        .fit_transform(ddf1, output_col='Embarked')\
        .drop(['PassengerId', 'Cabin', 'Ticket'])\
        .dropna(all_columns, how='any')\
        .replace({'male': 1, 'female': 0}, subset=['Sex'])\
        .map(title_checker, 'Name')\
        .map(age_categorizer, 'Age')\
        .map(fare_categorizer, 'Fare').cache()

    features.append('Name')
    ddf1 = StandardScaler(with_mean=True, with_std=True)\
        .fit_transform(ddf1, input_col=features, output_col=features)\
        .save("/titanic/titanic", storage='hdfs', host='master')

    compss_barrier()
    t3 = time.time()

    print("Time to preprocessing - t3-t2:", t3 - t2)

    # ddf1.show()
