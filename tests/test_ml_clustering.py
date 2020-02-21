#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from ddf_library.context import COMPSsContext
import pandas as pd


def base(cc):
    columns = ['x', 'y']
    n_samples = 1000
    from sklearn import datasets
    xy, labels = datasets.make_blobs(n_samples=n_samples)

    # df = pd.DataFrame([[1.0, 2.0], [1.0, 4.0], [1.0, 0], [4.0, 2.0],
    #                    [4.0, 4.0], [4.0, 0.0]], columns=cols)

    df = pd.DataFrame(xy, columns=columns)
    df['label'] = labels

    # creating DDF from a DataFrame
    ddf = cc.parallelize(df, 4)

    # scaling features using MinMax Scaler
    from ddf_library.functions.ml.feature import MinMaxScaler
    scaler = MinMaxScaler().fit(ddf, input_col=columns)
    ddf = scaler.transform(ddf)

    return ddf, columns


def kmeans(ddf, columns):

    from ddf_library.functions.ml.clustering import Kmeans
    clu = Kmeans(n_clusters=3, init_mode='k-means||')\
        .fit(ddf, feature_col=columns)
    clu.transform(ddf, pred_col='kmeans1').show(15)


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser(
            description="Testing Machine Learning Clustering")
    parser.add_argument('-o', '--operation',
                        type=int,
                        required=True,
                        help="""
                             1. kmeans
                            """)
    arg = vars(parser.parse_args())

    operation = arg['operation']
    list_operations = [kmeans]

    cc = COMPSsContext()
    ddf_test, cols = base(cc)

    list_operations[operation - 1](ddf_test, cols)
    cc.stop()
