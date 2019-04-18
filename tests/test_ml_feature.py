#!/usr/bin/python
# -*- coding: utf-8 -*-

from ddf_library.ddf import DDF
import pandas as pd
import numpy as np


def binarizer():
    print("\n_____Testing Binarizer_____\n")

    df = pd.DataFrame([[1., -1., 2.], [2., 0., 0.], [0., 1., -1.]],
                      columns=['x', 'y', 'z'])
    ddf = DDF().parallelize(df, 4)

    from ddf_library.functions.ml.feature import VectorAssembler, Binarizer
    assembler = VectorAssembler(input_col=["x", "y", 'z'],
                                output_col="features")
    ddf = assembler.transform(ddf)

    res = Binarizer(input_col='features', threshold=0)\
        .transform(ddf).to_df('features').tolist()

    sol = [[1., 0., 1.], [1., 0., 0.], [0., 1., 0.]]
    if not np.allclose(res, sol):
        raise Exception(" Output different from expected.")
    print("Ok")


def maxabs_scaler():
    print("_____Testing MaxAbsScaler _____")

    cols = ['x', 'y', 'z']
    df_maxabs = pd.DataFrame([[1., -1., 2.], [2., 0., 0.], [0., 1., -1.]],
                             columns=cols)

    # Creating DDF from DataFrame
    ddf_maxabs = DDF().parallelize(df_maxabs, 4)

    # Creating a column of features
    from ddf_library.functions.ml.feature import VectorAssembler
    assembler = VectorAssembler(input_col=["x", "y", 'z'],
                                output_col="features")
    ddf_maxabs = assembler.transform(ddf_maxabs)

    from ddf_library.functions.ml.feature import MaxAbsScaler
    ddf_maxabs = MaxAbsScaler(input_col=cols) \
        .fit_transform(ddf_maxabs)

    res = ddf_maxabs.to_df(cols).values.tolist()
    print(res)
    sol = [[0.5, -1., 1], [1.,  0., 0.], [0., 1., -0.5]]
    if not np.allclose(res, sol):
        raise Exception(" Output different from expected.")
    print("Ok")


def minmax_scaler():
    print("_____Testing MinMaxScaler _____")

    cols = ['x', 'y']
    df_minmax = pd.DataFrame([[-1, 2], [-0.5, 6], [0, 10], [1, 18]],
                             columns=cols)

    ddf_minmax = DDF().parallelize(df_minmax, 4)

    from ddf_library.functions.ml.feature import MinMaxScaler
    ddf_minmax = MinMaxScaler(input_col=cols)\
        .fit_transform(ddf_minmax)

    res = ddf_minmax.to_df(cols).values.tolist()
    sol = [[0., 0.], [0.25, 0.25], [0.5,  0.5], [1., 1.]]
    print(res)
    if not np.allclose(res, sol):
        raise Exception(" Output different from expected.")
    print("Ok")


def std_scaler():
    print("_____Testing StandardScaler _____")
    cols = ['x', 'y']
    df_std = pd.DataFrame([[0, 0], [0, 0], [1, 1], [1, 1]],
                          columns=cols)
    ddf_std = DDF().parallelize(df_std, 4)

    from ddf_library.functions.ml.feature import StandardScaler
    scaler = StandardScaler(input_col=cols, with_mean=True,
                            with_std=True).fit(ddf_std)
    ddf_std = scaler.transform(ddf_std)

    res = ddf_std.to_df(cols).values.tolist()
    print(res)
    sol = [[-1., -1.], [-1., -1.], [1., 1.], [1., 1.]]
    if not np.allclose(res, sol):
        raise Exception(" Output different from expected.")
    print("Ok")


def pca():

    print("\n_____Testing PCA_____\n")

    df = pd.read_csv('./iris-dataset.csv', sep=',')
    df.dropna(how="all", inplace=True)
    columns = df.columns.tolist()
    columns.remove('class')
    ddf = DDF().parallelize(df, 4)

    from ddf_library.functions.ml.feature import StandardScaler
    ddf_std = StandardScaler(input_col=columns).fit_transform(ddf)

    n_components = 2
    new_columns = ['col{}'.format(i) for i in range(n_components)]
    from ddf_library.functions.ml.feature import PCA
    pca = PCA(input_col=columns, output_col=new_columns,
              n_components=n_components)
    res = pca.fit_transform(ddf_std).to_df(new_columns).values.tolist()[0:10]

    sol = [[-2.26454173, -0.505703903],
           [-2.08642550,  0.655404729],
           [-2.36795045,  0.318477311],
           [-2.30419716,  0.575367713],
           [-2.38877749, -0.674767397],
           [-2.07053681, -1.51854856],
           [-2.44571134, -0.0745626750],
           [-2.23384186, -0.247613932],
           [-2.34195768,  1.09514636],
           [-2.18867576,  0.448629048]]
    sol_vals = [2.93035378,  0.92740362,  0.14834223,  0.02074601]
    if not np.allclose(res, sol) and \
            np.allclose(pca.model['eig_vals'], sol_vals):
        raise Exception(" Output different from expected.")
    print("Ok")


def poly_expansion():
    print("\n_____Testing PolynomialExpansion_____\n")

    df = pd.DataFrame([[0, 1], [2, 3], [4, 5]], columns=['x', 'y'])
    ddf = DDF().parallelize(df, 4)

    from ddf_library.functions.ml.feature import VectorAssembler
    assembler = VectorAssembler(input_col=["x", "y"], output_col="features")
    ddf = assembler.transform(ddf)

    from ddf_library.functions.ml.feature import PolynomialExpansion

    res = PolynomialExpansion(input_col='features', degree=2)\
        .transform(ddf).to_df('features').tolist()

    sol = [[1.0, 0.0, 1.0, 0.0, 0.0, 1.0],
           [1.0, 2.0, 3.0, 4.0, 6.0, 9.0],
           [1.0, 4.0, 5.0, 16.0, 20.0, 25.0]]
    if not np.allclose(res, sol):
        raise Exception(" Output different from expected.")
    print("Ok")


def onehot_encoder():
    print("\n_____Testing OneHotEncoder_____\n")

    from ddf_library.functions.ml.feature import OneHotEncoder
    df = pd.DataFrame([['Male', 1], ['Female', 3],
                       ['Female', 2]], columns=['x', 'y'])
    ddf = DDF().parallelize(df, 4)

    res = OneHotEncoder(input_col=['x', 'y'])\
        .fit_transform(ddf)\
        .to_df('features_onehot').tolist()

    sol = [[0.0, 1.0, 1.0, 0.0, 0.0],
           [1.0, 0.0, 0.0, 0.0, 1.0],
           [1.0, 0.0, 0.0, 1.0, 0.0]]
    if not np.allclose(res, sol):
        raise Exception(" Output different from expected.")
    print("Ok")


if __name__ == '__main__':
    print("_____Testing Features operations_____")

    # binarizer()
    pca()
    # poly_expansion()
    # onehot_encoder()
    # maxabs_scaler()
    # minmax_scaler()
    # std_scaler()
    #
