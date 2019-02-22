#!/usr/bin/python
# -*- coding: utf-8 -*-

from ddf.ddf import DDF
import pandas as pd


def ml_clustering():

    df = pd.DataFrame([[1.0, 2.0], [1.0, 4.0], [1.0, 0], [4.0, 2.0],
                       [4.0, 4.0], [4.0, 0.0]], columns=['x', 'y'])

    # creating DDF from a DataFrame
    ddf = DDF().parallelize(df, 4)

    from ddf.functions.ml.feature import VectorAssembler
    assembler = VectorAssembler(input_col=["x", "y"], output_col="features")
    ddf = assembler.transform(ddf).drop(['x', 'y'])

    # scaling features using MinMax Scaler
    from ddf.functions.ml.feature import MinMaxScaler
    scaler = MinMaxScaler(input_col='features', output_col='features_norm')\
        .fit(ddf)
    ddf = scaler.transform(ddf)

    from ddf.functions.ml.clustering import Kmeans
    kmeans = Kmeans(feature_col='features_norm', n_clusters=2,
                    init_mode='random', pred_col='kmeans1').fit(ddf)
    kmeans.save_model('/kmeans')
    ddf = kmeans.transform(ddf)

    # to test save and load models
    kmeans = Kmeans(feature_col='features_norm', n_clusters=2,
                    init_mode='random', pred_col='kmeans2')\
        .load_model('/kmeans')
    df = kmeans.transform(ddf).show(15)

    print "Kmeans result:\n", df
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
    print "_____Testing Machine Learning Clustering_____"
    ml_clustering()
