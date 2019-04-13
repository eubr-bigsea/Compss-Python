#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from ddf_library.ddf import DDF
import pandas as pd
from sklearn import datasets
import numpy as np


def simple_regression():
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

    df = pd.DataFrame.from_dict({'features': diabetes_x, 'y': diabetes_y})
    ddf_simple = DDF().parallelize(df, 4)

    from ddf_library.functions.ml.regression import OrdinaryLeastSquares
    model = OrdinaryLeastSquares('features', 'y').fit(ddf_simple)
    ddf_pred = model.transform(ddf_simple)

    sol_ddf = ddf_pred.to_df('pred_LinearReg')
    if not np.allclose(sol, sol_ddf):
        raise Exception("Wrong solution.")
    else:
        print("OK - Ordinary Least Squares.")

    return ddf_pred


def sgb_regression():
    print("\n_____SGB Regressor_____\n")

    diabetes = datasets.load_diabetes()

    # Use only one feature
    diabetes_x = diabetes.data.tolist()
    # diabetes_x = diabetes.data[:, 0: 4].tolist()
    diabetes_y = diabetes.target.tolist()
    df = pd.DataFrame.from_dict({'features': diabetes_x, 'y': diabetes_y})
    ddf = DDF().parallelize(df, 4)

    # Testing 'SGB' linear regressor
    ddf_train, ddf_test = ddf.split(0.7)

    from ddf_library.functions.ml.regression import SGDRegressor
    model = SGDRegressor('features', 'y', max_iter=20, alpha=1).fit(ddf_train)
    pred_ddf = model.transform(ddf_test)

    pred_ddf.show()

    return pred_ddf


def regressor_evaluator(ddf_pred):
    from ddf_library.functions.ml.evaluation import RegressionMetrics
    metrics = RegressionMetrics(col_features='features', label_col='y',
                                pred_col='pred_LinearReg', data=ddf_pred)

    print(metrics.get_metrics())


def evaluator_metrics():
    print("\n_____Regression Metrics Evaluator_____\n")
    data = pd.DataFrame([[14, 70, 2, 3.3490],
                         [16, 75, 5, 3.7180],
                         [27, 144, 7, 6.8472],
                         [42, 190, 9, 9.8400],
                         [39, 210, 10, 10.0151],
                         [50, 235, 13, 11.9783],
                         [83, 400, 20, 20.2529],
                         ], columns=['x', 'z', 'y', 'pred_LinearReg'])
    dataset = DDF().parallelize(data, 4)

    from ddf_library.functions.ml.feature import VectorAssembler
    assembler = VectorAssembler(input_col=["x"],
                                output_col="features")
    assembled = assembler.transform(dataset)

    from ddf_library.functions.ml.evaluation import RegressionMetrics
    metrics = RegressionMetrics(col_features='features', label_col='y',
                                pred_col='pred_LinearReg', data=assembled)

    print('Metrics for 2-D regression')
    sol = metrics.get_metrics()
    print(sol)
    """
                              Metric      Value
           R^2 (Explained Variance)    0.974235
           Mean Squared Error (MSE)    1.060066
     Root Mean Squared Error (RMSE)    1.029595
          Mean Absolute Error (MAE)    0.982700
    Mean Square of Regression (MSR)  200.413956
    """
    res = np.array([0.974235, 1.060066, 1.029595, 0.982700, 200.413956])
    if not np.allclose(res, sol['Value'].values):
        raise Exception('Result different from the expected.')

    assembler = VectorAssembler(input_col=["x", "z"],
                                output_col="features")
    assembled = assembler.transform(dataset)

    metrics = RegressionMetrics(col_features='features', label_col='y',
                                pred_col='pred_LinearReg', data=assembled)

    print('Metrics for 3-D regression')
    sol = metrics.get_metrics()
    print(sol)
    """
                              Metric      Value
           R^2 (Explained Variance)    0.974235
           Mean Squared Error (MSE)    1.325083
     Root Mean Squared Error (RMSE)    1.151122
          Mean Absolute Error (MAE)    1.228375
    Mean Square of Regression (MSR)  100.206978
    """
    res = np.array([0.974235, 1.325083, 1.151122, 1.228375, 100.206978])
    if not np.allclose(res, sol['Value'].values):
        raise Exception('Result different from the expected.')


if __name__ == '__main__':
    print("_____Testing Regressors_____")
    # pred_ddf = simple_regression()
    pred_ddf = sgb_regression()
    regressor_evaluator(pred_ddf)
    # evaluator_metrics()
