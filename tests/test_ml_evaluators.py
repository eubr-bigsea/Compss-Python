#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

from ddf_library.context import COMPSsContext


def binary_evaluator(cc):
    from ddf_library.functions.ml.evaluation import BinaryClassificationMetrics

    data = pd.DataFrame([[1, 70, 2, 0],
                         [1, 75, 5, 1],
                         [1, 144, 7, 1],
                         [0, 190, 9, 0],
                         [0, 210, 10, 0],
                         [0, 235, 13, 1],
                         [0, 400, 20, 1],
                         ], columns=['label', 'z', 'y', 'prediction'])
    dataset = cc.parallelize(data, 4)

    metrics_bin = BinaryClassificationMetrics(label_col='label',
                                              true_label=1,
                                              pred_col='prediction',
                                              ddf_var=dataset)

    # Get some metrics
    print("Binary Metrics:\n", metrics_bin.get_metrics(), '\n',
          metrics_bin.confusion_matrix)


def multi_label_evaluator(cc):
    from ddf_library.functions.ml.evaluation import MultilabelMetrics

    data = pd.DataFrame([[1, 70, 2, 1],
                         [1, 75, 5, 2],
                         [2, 144, 7, 2],
                         [3, 190, 9, 3],
                         [4, 210, 10, 4],
                         [1, 235, 13, 1],
                         [1, 400, 20, 1],
                         ], columns=['label', 'z', 'y', 'prediction'])
    dataset = cc.parallelize(data, 4)

    metrics_multi = MultilabelMetrics(label_col='label',
                                      pred_col='prediction',
                                      ddf_var=dataset)

    print("Multi label Metrics:\n",
          metrics_multi.get_metrics(), '\n',
          metrics_multi.confusion_matrix, '\n',
          metrics_multi.precision_recall)


def regressor_metrics(cc):
    print("\n_____Regression Metrics Evaluator_____\n")
    data = pd.DataFrame([[14, 70, 2, 3.3490],
                         [16, 75, 5, 3.7180],
                         [27, 144, 7, 6.8472],
                         [42, 190, 9, 9.8400],
                         [39, 210, 10, 10.0151],
                         [50, 235, 13, 11.9783],
                         [83, 400, 20, 20.2529],
                         ], columns=['x', 'z', 'y', 'prediction'])
    dataset = cc.parallelize(data, 4)

    from ddf_library.functions.ml.evaluation import RegressionMetrics
    metrics1 = RegressionMetrics(col_features=['x'], label_col='y',
                                 pred_col='prediction', data=dataset)

    print('Metrics for 2-D regression')
    sol1 = metrics1.get_metrics()
    print(sol1)
    """
                              Metric      Value
           R^2 (Explained Variance)    0.974235
           Mean Squared Error (MSE)    1.060066
     Root Mean Squared Error (RMSE)    1.029595
          Mean Absolute Error (MAE)    0.982700
    Mean Square of Regression (MSR)  200.413956
    """
    res = np.array([0.974235, 1.060066, 1.029595, 0.982700, 200.413956])
    if not np.allclose(res, sol1['Value'].values):
        raise Exception('Result different from the expected.')

    metrics2 = RegressionMetrics(col_features=["x", "z"], label_col='y',
                                 pred_col='prediction', data=dataset)

    print('Metrics for 3-D regression')
    sol2 = metrics2.get_metrics()
    print(sol2)
    """
                              Metric      Value
           R^2 (Explained Variance)    0.974235
           Mean Squared Error (MSE)    1.325083
     Root Mean Squared Error (RMSE)    1.151122
          Mean Absolute Error (MAE)    1.228375
    Mean Square of Regression (MSR)  100.206978
    """
    res = np.array([0.974235, 1.325083, 1.151122, 1.228375, 100.206978])
    if not np.allclose(res, sol2['Value'].values):
        raise Exception('Result different from the expected.')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
            description="Testing Machine Learning Clustering")
    parser.add_argument('-o', '--operation',
                        type=int,
                        required=True,
                        help="""
                             1. Binary evaluator (classification)
                             2. Multi label evaluator (classification)
                             3. Regressor metrics (regression)
                            """)
    arg = vars(parser.parse_args())

    operation = arg['operation']
    list_operations = [binary_evaluator,
                       multi_label_evaluator,
                       regressor_metrics]
    cc = COMPSsContext()
    list_operations[operation - 1](cc)
    cc.stop()
