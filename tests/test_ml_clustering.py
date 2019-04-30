#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from ddf_library.ddf import DDF
import pandas as pd


def base():
    cols = ['x', 'y']
    n_samples = 1000
    from sklearn import datasets
    xy, labels = datasets.make_blobs(n_samples=n_samples)

    # df = pd.DataFrame([[1.0, 2.0], [1.0, 4.0], [1.0, 0], [4.0, 2.0],
    #                    [4.0, 4.0], [4.0, 0.0]], columns=cols)

    df = pd.DataFrame(xy, columns=cols)
    df['label'] = labels

    # creating DDF from a DataFrame
    ddf = DDF().parallelize(df, 4)

    # scaling features using MinMax Scaler
    from ddf_library.functions.ml.feature import MinMaxScaler
    scaler = MinMaxScaler(input_col=cols).fit(ddf)
    ddf = scaler.transform(ddf)

    return ddf, cols


def kmeans(ddf, cols):

    from ddf_library.functions.ml.clustering import Kmeans
    kmeans = Kmeans(feature_col=cols, n_clusters=3,
                    init_mode='k-means||').fit(ddf)
    kmeans.save_model('/kmeans')
    del kmeans

    # to test save and load models
    kmeans = Kmeans(feature_col=cols, n_clusters=3,
                    init_mode='k-means||')\
        .load_model('/kmeans')
    print(kmeans.model)
    kmeans.transform(ddf, pred_col='kmeans1').show(15)

    """
               features features_norm  kmeans1  kmeans2
        0  [1.0, 2.0]    [0.0, 0.5]      0.0      0.0
        1  [1.0, 4.0]    [0.0, 1.0]      0.0      0.0
        2  [1.0, 0.0]    [0.0, 0.0]      0.0      0.0
        3  [4.0, 2.0]    [1.0, 0.5]      1.0      1.0
        4  [4.0, 4.0]    [1.0, 1.0]      1.0      1.0
        5  [4.0, 0.0]    [1.0, 0.0]      1.0      1.0  
    """


if __name__ == '__main__':
    print("_____Testing Machine Learning Clustering_____")
    ddf_test, cols = base()
    kmeans(ddf_test, cols)
