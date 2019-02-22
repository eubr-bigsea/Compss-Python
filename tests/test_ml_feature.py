#!/usr/bin/python
# -*- coding: utf-8 -*-

from ddf.ddf import DDF
import pandas as pd

def ml_feature_scalers():
    print "_____Testing Feature Scalers_____"

    df_maxabs = pd.DataFrame([[1., -1., 2.], [2., 0., 0.], [0., 1., -1.]],
                             columns=['x', 'y', 'z'])
    df_minmax = pd.DataFrame([[-1, 2], [-0.5, 6], [0, 10], [1, 18]],
                             columns=['x', 'y'])
    df_std = pd.DataFrame([[0, 0], [0, 0], [1, 1], [1, 1]],
                          columns=['x', 'y'])

    # Creating DDF from DataFrame
    ddf_maxabs = DDF().parallelize(df_maxabs, 4)
    ddf_minmax = DDF().parallelize(df_minmax, 4)
    ddf_std = DDF().parallelize(df_std, 4)

    # Creating a column of features
    from ddf.functions.ml.feature import VectorAssembler
    assembler = VectorAssembler(input_col=["x", "y", 'z'],
                                output_col="features")
    ddf_maxabs = assembler.transform(ddf_maxabs)

    assembler = VectorAssembler(input_col=["x", "y"], output_col="features")
    ddf_minmax = assembler.transform(ddf_minmax)
    ddf_std = assembler.transform(ddf_std)

    from ddf.functions.ml.feature import MaxAbsScaler
    ddf_maxabs = MaxAbsScaler(input_col='features', output_col='features_norm')\
        .fit_transform(ddf_maxabs).select(['features_norm'])

    from ddf.functions.ml.feature import MinMaxScaler
    ddf_minmax = MinMaxScaler(input_col='features', output_col='features_norm')\
        .fit_transform(ddf_minmax).select(['features_norm'])

    from ddf.functions.ml.feature import StandardScaler
    scaler = StandardScaler(input_col='features', output_col='features_norm',
                            with_mean=True, with_std=True).fit(ddf_std)
    ddf_std = scaler.transform(ddf_std).select(['features_norm'])

    print "MaxAbsScaler :\n", ddf_maxabs.show()
    """
    [[0.5, -1.,  1.],
     [1.,   0.,  0.],
     [0.,   1., -0.5]]
    """

    print "\nMinMaxScaler :\n", ddf_minmax.show()
    """
    [[0.   0.]
     [0.25 0.25]
     [0.5  0.5]
     [1.   1.]]
    """

    print "\nStandardScaler :\n", ddf_std.show()
    """
    [[-1. - 1.]
     [-1. - 1.]
     [1.    1.]
     [1.    1.]]
    """


def ml_feature_dimensionality():

    print "\n_____Testing PCA_____\n"

    df = pd.read_csv('tests/iris-dataset.csv', sep=',')
    df.dropna(how="all", inplace=True)
    columns = df.columns.tolist()
    columns.remove('class')

    ddf = DDF().parallelize(df, 4)
    from ddf.functions.ml.feature import VectorAssembler
    assembler = VectorAssembler(input_col=columns, output_col="features")
    ddf = assembler.transform(ddf)

    from ddf.functions.ml.feature import StandardScaler
    ddf_std = StandardScaler(input_col='features',
                             output_col='features_norm').fit_transform(ddf)

    from ddf.functions.ml.feature import PCA
    pca = PCA(input_col='features_norm', output_col='features_pca',
              n_components=2)
    ddf_pca = pca.fit_transform(ddf_std).select(['features', 'features_pca'])
    print "Eigenvectors:\n", pca.model['eig_vecs']
    print "Eigenvalues:\n", pca.model['eig_vals']
    """
    Eigenvectors:
    [[ 0.52237162 -0.37231836 -0.72101681  0.26199559]
     [-0.26335492 -0.92555649  0.24203288 -0.12413481]
     [ 0.58125401 -0.02109478  0.14089226 -0.80115427]
     [ 0.56561105 -0.06541577  0.6338014   0.52354627]]
    
    Eigenvalues:
    [ 2.93035378  0.92740362  0.14834223  0.02074601]
    """

    print "PCA output :\n", ddf_pca.show()
    """
    [[-2.26454173 -0.505703903]
     [-2.08642550  0.655404729]
     [-2.36795045  0.318477311]
     [-2.30419716  0.575367713]
     [-2.38877749 -0.674767397]
     [-2.07053681 -1.51854856]
     [-2.44571134 -0.0745626750]
     [-2.23384186 -0.247613932]
     [-2.34195768  1.09514636]
     [-2.18867576  0.448629048]
     [-2.16348656 -1.07059558]
     [-2.32737775 -0.158587455]
     [-2.22408272  0.709118158]
     [-2.63971626  0.938281982]
     [-2.19229151 -1.88997851]
     [-2.25146521 -2.72237108]
     [-2.20275048 -1.51375028]
     [-2.19017916 -0.514304308]
     [-1.89407429 -1.43111071]
     [-2.33994907 -1.15803343] 
    """


if __name__ == '__main__':

    # ml_feature_scalers()
    ml_feature_dimensionality()
