#!/usr/bin/python
# -*- coding: utf-8 -*-

from ddf.ddf import DDF
import pandas as pd


def ml_regression_simple():
    print "\n_____Simple Regressor_____\n"

    # Testing 'simple' linear
    from sklearn import datasets
    import numpy as np
    diabetes = datasets.load_diabetes()

    # Use only one feature
    diabetes_x = diabetes.data[:, np.newaxis, 2].tolist()
    diabetes_y = diabetes.target.tolist()

    df = pd.DataFrame.from_dict({'features': diabetes_x, 'y': diabetes_y})
    ddf_simple = DDF().parallelize(df, 4)

    from ddf.functions.ml.feature import StandardScaler
    scaler = StandardScaler(input_col='features',
                            output_col='features').fit(ddf_simple)
    ddf_simple = scaler.transform(ddf_simple)

    ddf_train, ddf_test = ddf_simple.split(0.5)

    from ddf.functions.ml.regression import LinearRegression
    model = LinearRegression('features', 'y', mode='simple').fit(ddf_train)
    ddf_test = model.transform(ddf_test)

    print "Simple linear regressor result:\n", ddf_test.show(10)

    from ddf.functions.ml.evaluation import RegressionMetrics
    metrics = RegressionMetrics(col_features='features', label_col='y',
                                pred_col='pred_LinearReg', data=ddf_test)

    print metrics.get_metrics()


def ml_regression_metrics():
    print "\n_____Regression Metrics Evaluator_____\n"
    data = pd.DataFrame([[14, 70, 2, 3.3490],
                         [16, 75, 5, 3.7180],
                         [27, 144, 7, 6.8472],
                         [42, 190, 9, 9.8400],
                         [39, 210, 10, 10.0151],
                         [50, 235, 13, 11.9783],
                         [83, 400, 20, 20.2529],
                         ], columns=['x', 'z', 'y', 'pred_LinearReg'])
    dataset = DDF().parallelize(data, 4)

    from ddf.functions.ml.feature import VectorAssembler
    assembler = VectorAssembler(input_col=["x"],
                                output_col="features")
    assembled = assembler.transform(dataset)

    from ddf.functions.ml.evaluation import RegressionMetrics
    metrics = RegressionMetrics(col_features='features', label_col='y',
                                pred_col='pred_LinearReg', data=assembled)

    print 'Metrics for 2-D regression'
    print metrics.get_metrics()
    """
                              Metric      Value
           R^2 (Explained Variance)    0.974235
           Mean Squared Error (MSE)    1.060066
     Root Mean Squared Error (RMSE)    1.029595
          Mean Absolute Error (MAE)    0.982700
    Mean Square of Regression (MSR)  200.413956
    """

    assembler = VectorAssembler(input_col=["x", "z"],
                                output_col="features")
    assembled = assembler.transform(dataset)

    metrics = RegressionMetrics(col_features='features', label_col='y',
                                pred_col='pred_LinearReg', data=assembled)

    print 'Metrics for 3-D regression'
    print metrics.get_metrics()
    """
                              Metric      Value
           R^2 (Explained Variance)    0.974235
           Mean Squared Error (MSE)    1.325083
     Root Mean Squared Error (RMSE)    1.151122
          Mean Absolute Error (MAE)    1.228375
    Mean Square of Regression (MSR)  100.206978
    """


def ml_regression_sgb():
    print "\n_____SGB Regressor_____\n"
    df = pd.DataFrame([[14, 70, 2, 3.3490],
                      [16, 75, 5, 3.7180],
                      [27, 144, 7, 6.8472],
                      [42, 190, 9, 9.8400],
                      [39, 210, 10, 10.0151],
                      [50, 235, 13, 11.9783],
                      [83, 400, 20, 20.2529]],
                      columns=['x', 'z', 'y', 'result1'])
    ddf = DDF().parallelize(df, 4)

    # Testing 'SGB' linear regressor

    from ddf.functions.ml.feature import VectorAssembler
    assembler = VectorAssembler(input_col=["x", "z"], output_col="features")
    ddf = assembler.transform(ddf)

    from ddf.functions.ml.feature import MinMaxScaler
    scaler = MinMaxScaler(input_col='features',
                          output_col='features').fit(ddf)
    ddf = scaler.transform(ddf)

    from ddf.functions.ml.regression import LinearRegression
    model = LinearRegression('features', 'y', max_iter=20, alpha=0.01).fit(ddf)
    df = model.transform(ddf).show()

    print "Result:\n", df


if __name__ == '__main__':
    print "_____Testing Machine Learning Regressors_____"
    ml_regression_simple()
    # ml_regression_metrics()
    #ml_regression_sgb()
