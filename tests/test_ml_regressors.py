#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from ddf_library.context import COMPSsContext
import pandas as pd
from sklearn import datasets
import numpy as np


def simple_regression(cc):
    print("\n_____Ordinary Least Squares Regressor_____\n")

    # Testing 'simple' linear

    diabetes = datasets.load_diabetes()

    # Use only one feature
    diabetes_x = diabetes.data[:, np.newaxis, 2].tolist()
    diabetes_y = diabetes.target.tolist()

    # to compare
    from sklearn import linear_model
    clf = linear_model.LinearRegression()
    clf.fit(diabetes_x, diabetes_y)
    sol = clf.predict(diabetes_x)

    diabetes_x = np.array(diabetes_x).flatten()
    df = pd.DataFrame.from_dict({'features': diabetes_x,
                                 'y': diabetes_y,
                                 'sol': sol})

    ddf_simple = cc.parallelize(df, 4)

    from ddf_library.functions.ml.regression import OrdinaryLeastSquares
    model = OrdinaryLeastSquares().fit(ddf_simple, 'features', 'y')
    ddf_pred = model.transform(ddf_simple, pred_col='pred_LinearReg')

    sol_ddf = ddf_pred.to_df('pred_LinearReg').values.flatten()
    if not np.allclose(sol, sol_ddf):
        raise Exception("Wrong solution.")

    else:
        print("OK - Ordinary Least Squares.")


def sgb_regression(cc):
    print("\n_____SGB Regressor_____\n")

    diabetes = datasets.load_diabetes()

    # Use only one feature
    # diabetes_x = diabetes.data.tolist()
    diabetes_x = diabetes.data[:, 0: 4].tolist()
    diabetes_y = diabetes.target.tolist()

    cols = ['col{}'.format(i) for i in range(len(diabetes_x[0]))]
    df = pd.DataFrame(diabetes_x, columns=cols)
    df['y'] = diabetes_y

    ddf = cc.parallelize(df, 4)
    # Testing 'SGB' linear regressor
    ddf_train, ddf_test = ddf.split(0.7)

    from ddf_library.functions.ml.regression import GDRegressor
    model = GDRegressor(max_iter=15, alpha=1).fit(ddf_train, cols, 'y')
    pred_ddf = model.transform(ddf_test)
    pred_ddf.to_df()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
            description="Testing Regressor algorithms")
    parser.add_argument('-o', '--operation',
                        type=int,
                        required=True,
                        help="""
                        1. simple regression (2D)
                        2. SGB regression (2D+)
                        """)
    arg = vars(parser.parse_args())

    operation = arg['operation']
    list_operations = [simple_regression, sgb_regression]
    cc = COMPSsContext()
    list_operations[operation - 1](cc)
    cc.stop()
