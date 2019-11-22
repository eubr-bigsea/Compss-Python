#!/usr/bin/python
# -*- coding: utf-8 -*-

from ddf_library.ddf import DDF
import pandas as pd
import numpy as np


def binarizer():
    print("\n_____Testing Binarizer_____\n")

    columns = ['x', 'y', 'z']
    df = pd.DataFrame([[1., -1., 2.], [2., 0., 0.], [0., 1., -1.]],
                      columns=columns)
    ddf = DDF().parallelize(df, 4)

    from ddf_library.functions.ml.feature import Binarizer

    res = Binarizer(input_col=columns, threshold=0, remove=True)\
        .transform(ddf, output_col=columns).to_df().values.tolist()

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

    from ddf_library.functions.ml.feature import MaxAbsScaler
    ddf_maxabs = MaxAbsScaler() \
        .fit_transform(ddf_maxabs, input_col=cols)

    res = ddf_maxabs.to_df(cols).values.tolist()

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
    ddf_minmax = MinMaxScaler()\
        .fit_transform(ddf_minmax, input_col=cols)

    res = ddf_minmax.to_df(cols).values.tolist()
    sol = [[0., 0.], [0.25, 0.25], [0.5,  0.5], [1., 1.]]

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
    scaler = StandardScaler(with_mean=True,
                            with_std=True).fit(ddf_std, input_col=cols)
    ddf_std = scaler.transform(ddf_std)

    res = ddf_std.to_df(cols).values.tolist()
    sol = [[-1., -1.], [-1., -1.], [1., 1.], [1., 1.]]
    if not np.allclose(res, sol):
        raise Exception(" Output different from expected.")
    print("Ok")


def pca_workflow():

    print("\n_____Testing PCA_____\n")

    df = pd.read_csv('./iris-dataset.csv', sep=',')
    df.dropna(how="all", inplace=True)
    columns = df.columns.tolist()
    columns.remove('class')
    ddf = DDF().parallelize(df, 4)

    from ddf_library.functions.ml.feature import StandardScaler
    ddf_std = StandardScaler().fit_transform(ddf, input_col=columns)

    n_components = 2
    new_columns = ['col{}'.format(i) for i in range(n_components)]
    from ddf_library.functions.ml.feature import PCA
    pca = PCA(input_col=columns, n_components=n_components)
    res = pca.fit_transform(ddf_std, output_col=new_columns)\
             .to_df(new_columns).values.tolist()[0:10]

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

    sol_values = [2.93035378,  0.92740362,  0.14834223,  0.02074601]
    cond1 = np.allclose(res, sol)
    cond2 = np.allclose(pca.model['eig_values'], sol_values)
    if not all([cond1, cond2]):
        raise Exception(" Output different from expected.")
    print("Ok")


def poly_expansion():
    print("\n_____Testing PolynomialExpansion_____\n")

    columns = ["x", "y"]
    df = pd.DataFrame([[0, 1], [2, 3], [4, 5]], columns=columns)
    ddf = DDF().parallelize(df, 4)

    from ddf_library.functions.ml.feature import PolynomialExpansion

    res = PolynomialExpansion(input_col=columns, degree=2, remove=True)\
        .transform(ddf).to_df().values.tolist()

    sol = [[1.0, 0.0, 1.0, 0.0, 0.0, 1.0],
           [1.0, 2.0, 3.0, 4.0, 6.0, 9.0],
           [1.0, 4.0, 5.0, 16.0, 20.0, 25.0]]
    if not np.allclose(res, sol):
        raise Exception(" Output different from expected.")
    print("Ok")


def onehot_encoder():
    print("\n_____Testing OneHotEncoder_____\n")

    columns = ['x', 'y']
    from ddf_library.functions.ml.feature import OneHotEncoder
    df = pd.DataFrame([['Male', 1], ['Female', 3],
                       ['Female', 2]], columns=columns)
    ddf = DDF().parallelize(df, 4)

    res = OneHotEncoder(input_col=columns, remove=True)\
        .fit_transform(ddf, output_col='_1hot')\
        .to_df().values.tolist()

    sol = [[0.0, 1.0, 1.0, 0.0, 0.0],
           [1.0, 0.0, 0.0, 0.0, 1.0],
           [1.0, 0.0, 0.0, 1.0, 0.0]]
    if not np.allclose(res, sol):
        raise Exception(" Output different from expected.")
    print("Ok")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description="Testing Features operations")
    parser.add_argument('-o', '--operation',
                        type=int,
                        required=True,
                        help="""
                         1. binarizer
                         2. PCA workflow
                         3. polynomial expansion
                         4. One hot encoder
                         5. Maxabs scaler
                         6. Minmax scaler
                         7. Standard scaler
                        """)
    arg = vars(parser.parse_args())

    operation = arg['operation']
    list_operations = [binarizer, pca_workflow,
                       poly_expansion, onehot_encoder,
                       maxabs_scaler, minmax_scaler, std_scaler]
    list_operations[operation-1]()
