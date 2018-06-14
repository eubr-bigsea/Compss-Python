#!/usr/bin/python
# -*- coding: utf-8 -*-
"""PCA.

Principal component analysis (PCA) is a statistical method to find
a rotation such that the first coordinate has the largest variance
possible, and each succeeding coordinate in turn has the largest
variance possible. The columns of the rotation matrix are called
principal components. PCA is used widely in dimensionality reduction.
"""
__author__ = "Lucas Miguel S Ponce"
__email__ = "lucasmsp@gmail.com"

from pycompss.api.task import task
from pycompss.functions.reduce import mergeReduce
#from pycompss.api.local import local
import numpy as np


class PCA(object):
    """PCA's methods.

    - fit()
    - transform()
    """

    def fit(self, data, settings, nfrag):
        """Fit.

        - :param data:        A list with nfrag pandas's dataframe
                              used to create the model.
        - :param settings:    A dictionary that contains:
            - features: 	  Field of the features in the dataset;
            - NComponents     Number of wanted dimensionality
                              (int, 0 < NComponents <= dim(features)).
        - :param nfrag:     A number of fragments;
        - :return:            Returns a model (which is a pandas dataframe).
        """
        if 'NComponents' not in settings:
            raise Exception('You must inform a Number of Components.')

        if 'features' not in settings:
            raise Exception('You must inform a valid `features` column.')

        NComponents = settings.get('NComponents')
        cols = settings.get('features')

        partial_count = [[] for _ in range(nfrag)]
        for f in range(nfrag):
            partial_count[f] = pca_count(data[f], cols)

        mergedCount = mergeReduce(pca_mergeCount, partial_count)
        mergedCount = meanCalc(mergedCount)
        for f in range(nfrag):
            partial_count[f] = partial_multiply(data[f], cols, mergedCount)

        mergedcov = mergeReduce(pca_mergeCov, partial_count)

        info = pca_eigendecomposition(mergedcov, mergedCount, NComponents)
        from pycompss.api.api import compss_wait_on
        info = compss_wait_on(info)

        model = dict()
        model['algorithm'] = 'PCA'
        model['cum_var_exp'] = info[0]
        model['eig_vals'] = info[1]
        model['eig_vecs'] = info[2]
        model['model'] = info[3]
        return model

    def transform(self, data, model, settings, nfrag):
        """Transform.

        - :param data:      A list with nfrag pandas's dataframe.
        - :param model:		The pca model created;
        - :param settings:  A dictionary that contains:
            - features: 	Field of the features data;
            - predCol:    	Alias to the new features field;
        - :param nfrag:   A number of fragments;
        - :return:          A list with nfrag pandas's dataframe
                            (in the same input format).
        """
        if 'features' not in settings:
            raise Exception("You must inform the `features` field.")

        if model.get('algorithm', 'null') != 'PCA':
                raise Exception("You must inform a valid model.")

        model = model['model']
        features_col = settings['features']
        predCol = settings.get('predCol',
                               '{}_reduced'.format(features_col))

        result = [[] for _ in range(nfrag)]
        for f in range(nfrag):
            result[f] = _pca_transform(data[f], features_col, predCol, model)

        return result

    def transform_serial(self, data, model, settings):
        """Transform.

        - :param data:      A list with nfrag pandas's dataframe.
        - :param model:		The pca model created;
        - :param settings:  A dictionary that contains:
            - features: 	Field of the features data;
            - predCol:    	Alias to the new features field;
        - :param nfrag:   A number of fragments;
        - :return:          A list with nfrag pandas's dataframe
                            (in the same input format).
        """
        if 'features' not in settings:
            raise Exception("You must inform the `features` field.")

        if model.get('algorithm', 'null') != 'PCA':
                raise Exception("You must inform a valid model.")

        model = model['model']
        features_col = settings['features']
        predCol = settings.get('predCol',
                               '{}_reduced'.format(features_col))

        return _pca_transform_(data, features_col, predCol, model)


@task(returns=list)
def pca_count(data, cols):
    """Partial count."""
    N = len(data)
    partialsum = 0
    if N > 0:
        data = data[cols].values
        partialsum = reduce(lambda l1, l2: np.add(l1, l2), data)
    return [N, partialsum]


@task(returns=list)
def pca_mergeCount(count1, count2):
    """Merge partial counts."""
    N = count1[0] + count2[0]
    partialsum = np.add(count1[1], count2[1])
    return [N, partialsum]


# @local
@task(returns=list, priority=True)
def meanCalc(mergedCount):
    """Generate the mean value."""
    # This method could be executed nfrag times inside each next function
    # with this, we can remove this function

    N = mergedCount[0]
    mean = np.array(map(lambda x: float(x)/N, mergedCount[1]))
    return [mean, N]


@task(returns=list)
def partial_multiply(data, col, mean):
    """Perform partial calculation."""
    cov_mat = 0

    if len(data) > 0:
        mean_vec = mean[0]
        X_std = data[col].values
        X_std = np.array(X_std.tolist())

        first_part = X_std - mean_vec
        cov_mat = first_part.T.dot(first_part)

    return cov_mat


@task(returns=list)
def pca_mergeCov(cov1, cov2):
    """Merge covariance."""
    return np.add(cov1, cov2)


# @local
@task(returns=list, priority=True)
def pca_eigendecomposition(cov_mat, mean, NComponents):
    """Generate a eigen decomposition."""
    N = mean[1]
    M = len(cov_mat)
    cov_mat = cov_mat / (N-1)
    eig_vals, eig_vecs = np.linalg.eig(cov_mat)

    # Sort the eigenvalue and vecs tuples from high to low
    inds = eig_vals.argsort()[::-1]
    sortedEighVals = eig_vals[inds]
    sortedEighVecs = eig_vecs[inds]

    tot = sum(eig_vals)
    var_exp = [(i / tot)*100 for i in sortedEighVals]
    cum_var_exp = np.cumsum(var_exp)

    NComponents = min([NComponents, M])

    matrix_w = sortedEighVecs[:, 0:NComponents]

    return [cum_var_exp, sortedEighVals, sortedEighVecs, matrix_w]


@task(returns=list)
def _pca_transform(data, features, predCol, model):
    """Reduce the dimensionality based in the created model."""
    return _pca_transform_(data, features, predCol, model)


def _pca_transform_(data, features, predCol, model):
    """Reduce the dimensionality based in the created model."""
    tmp = []
    if len(data) > 0:
        tmp = np.array(data[features].values.tolist()).dot(model).tolist()

    data[predCol] = tmp
    return data