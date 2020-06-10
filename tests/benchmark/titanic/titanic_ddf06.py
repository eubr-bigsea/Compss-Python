#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pycompss.api.task import task
from pycompss.api.api import compss_barrier

from ddf_library.context import COMPSsContext
from ddf_library.utils import generate_info
from ddf_library.functions.ml.feature import StringIndexer, StandardScaler
from ddf_library.columns import col, udf
from ddf_library.types import DataType

import pandas as pd
import numpy as np
import time
import sys
import math
from io import StringIO

####
#   Data generation
#### 


@task(returns=2)
def generate_partition(x, multiplier, frag):
    x = x * multiplier
    cols = ['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age',
            'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']
    dtypes = {'PassengerId': np.int16, 'Survived': np.int8, 'Pclass': np.int8,
              'Name': np.dtype('O'), 'Sex': np.dtype('O'), 'Age': np.float32,
              'SibSp': np.int8, 'Parch': np.int8, 'Ticket': np.dtype('O'),
              'Fare': np.float32, 'Cabin': np.dtype('O'),
              'Embarked': np.dtype('O')}
    df = pd.read_csv(StringIO(x), sep=',',
                     names=cols, header=None, dtype=dtypes)
    schema = generate_info(df, frag)
    return df, schema


def generate_data(n_repeat, nfrag):
    dfs = [[] for _ in range(nfrag)]
    info = [[] for _ in range(nfrag)]
    block = ''
    with open('base_titanic.csv', 'r') as content_file:
        block = content_file.read()

    for f in range(nfrag):
        dfs[f], info[f] = generate_partition(block, n_repeat, f)
    return dfs, info


def titanic(ddf1):

    def title_checker(name):
        titles = {"Mr.": 1, "Miss": 2, "Mrs.": 3, "Master": 4, "Rare": 5}
        for title in titles:
            if title in name:
                return titles[title]
        return -1

    def age_categorizer(age):
        category = 7
        if age <= 11:
            category = 0
        elif (age > 11) and (age <= 18):
            category = 1
        elif (age > 18) and (age <= 22):
            category = 2
        elif (age > 22) and (age <= 27):
            category = 3
        elif (age > 27) and (age <= 33):
            category = 4
        elif (age > 33) and (age <= 40):
            category = 5
        elif (age > 40) and (age <= 66):
            category = 6

        return category

    def fare_categorizer(fare):
        category = 5
        if fare <= 7.91:
            category = 0
        elif (fare > 7.91) and (fare <= 14.454):
            category = 1
        elif (fare > 14.454) and (fare <= 31):
            category = 2
        elif (fare > 31) and (fare <= 99):
            category = 3
        elif (fare > 99) and (fare <= 250):
            category = 4
        return category

    title_checker_udf = udf(title_checker, DataType.INT, col('Name'))
    age_categorizer_udf = udf(age_categorizer, DataType.INT, col('Age'))
    fare_categorizer_udf = udf(fare_categorizer, DataType.INT, col('Fare'))

    features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
    all_columns = features + ['Survived']

    ddf2 = StringIndexer() \
        .fit_transform(ddf1, input_col=['Embarked'], output_col=['Embarked']) \
        .drop(['PassengerId', 'Cabin', 'Ticket']) \
        .dropna(all_columns, how='any') \
        .replace({'male': 1, 'female': 0}, subset=['Sex']) \
        .map(title_checker_udf, 'Name') \
        .map(age_categorizer_udf, 'Age') \
        .map(fare_categorizer_udf, 'Fare')  # .cache()

    features.append('Name')
    ddf3 = StandardScaler(with_mean=True, with_std=True) \
        .fit_transform(ddf2, input_col=features, output_col=features)
    ddf3.save.csv("hdfs://localhost:9000/titanic-out", mode='overwrite')


if __name__ == "__main__":

    print("Titanic workflow implemented to DDF v0.6")

    size_block = int(sys.argv[1]) * 1024 * 1024  # in bytes,
    n_frag = int(sys.argv[2])
    artificial_data = size_block > 0
    cc = COMPSsContext()

    if artificial_data:
        # if working with artificial data:
        size_original = 61113
        n_repeat = int(math.ceil(size_block / size_original))
        ddf1 = cc.import_compss_data(generate_data(n_repeat, n_frag))
    else:

        schema = {'PassengerId': DataType.INT, 'Survived': DataType.INT,
                  'Pclass': DataType.INT, 'Name': DataType.STRING,
                  'Sex': DataType.STRING, 'Age': DataType.DECIMAL,
                  'SibSp': DataType.INT, 'Parch': DataType.INT,
                  'Ticket': DataType.STRING, 'Fare': DataType.DECIMAL,
                  'Cabin': DataType.STRING, 'Embarked': DataType.STRING}
        cc = COMPSsContext()
        ddf1 = cc.read.csv("hdfs://localhost:9000/titanic.csv",
                           schema=schema, sep='\t', num_of_parts=n_frag)

    titanic(ddf1)

    cc.stop()
