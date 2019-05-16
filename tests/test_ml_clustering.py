#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from ddf_library.ddf import DDF
import pandas as pd


def base():
    columns = ['x', 'y']
    n_samples = 1000
    from sklearn import datasets
    xy, labels = datasets.make_blobs(n_samples=n_samples)

    # df = pd.DataFrame([[1.0, 2.0], [1.0, 4.0], [1.0, 0], [4.0, 2.0],
    #                    [4.0, 4.0], [4.0, 0.0]], columns=cols)

    df = pd.DataFrame(xy, columns=columns)
    df['label'] = labels

    # creating DDF from a DataFrame
    ddf = DDF().parallelize(df, 4)

    # scaling features using MinMax Scaler
    from ddf_library.functions.ml.feature import MinMaxScaler
    scaler = MinMaxScaler(input_col=columns).fit(ddf)
    ddf = scaler.transform(ddf)

    return ddf, columns


def kmeans(ddf, columns):

    from ddf_library.functions.ml.clustering import Kmeans
    clu = Kmeans(feature_col=columns, n_clusters=3,
                 init_mode='k-means||').fit(ddf)
    clu.save_model('/kmeans')
    del clu

    # to test save and load models
    clu = Kmeans(feature_col=cols, n_clusters=3, init_mode='k-means||')\
        .load_model('/kmeans')
    print(clu.model)
    clu.transform(ddf, pred_col='kmeans1').show(15)


if __name__ == '__main__':
    print("_____Testing Machine Learning Clustering_____")
    ddf_test, cols = base()
    kmeans(ddf_test, cols)
