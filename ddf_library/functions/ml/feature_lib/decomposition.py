#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Lucas Miguel S Ponce"
__email__ = "lucasmsp@gmail.com"

from pycompss.api.task import task
from pycompss.functions.reduce import merge_reduce
from pycompss.api.api import compss_wait_on, compss_delete_object
# from pycompss.api.local import *  # requires guppy

from ddf_library.ddf import DDF, DDFSketch
from ddf_library.ddf_model import ModelDDF
from ddf_library.utils import generate_info

import numpy as np
import pandas as pd


class PCA(ModelDDF):
    """
    Principal component analysis (PCA) is a statistical method to find
    a rotation such that the first coordinate has the largest variance
    possible, and each succeeding coordinate in turn has the largest
    variance possible. The columns of the rotation matrix are called
    principal components. PCA is used widely in dimensionality reduction.

    :Example:

    >>> pca = PCA(input_col='features', output_col='features_pca',
    >>>           n_components=2).fit(ddf1)
    >>> ddf2 = pca.transform(ddf1)
    """

    def __init__(self, input_col, output_col, n_components, remove=False):
        """
        :param input_col: Input feature column;
        :param n_components: Number of output components;
        :param output_col: A list of output feature column or a suffix name.
        """
        super(PCA, self).__init__()

        if not isinstance(input_col, list):
            input_col = [input_col]

        if not isinstance(output_col, list):
            output_col = ['{}{}'.format(col, output_col) for col in input_col]

        self.settings = dict()
        self.settings['input_col'] = input_col
        self.settings['output_col'] = output_col
        self.settings['n_components'] = n_components
        self.settings['remove'] = remove

        self.name = 'PCA'
        self.var_exp = self.cum_var_exp = \
            self.eig_vals = self.eig_vecs = self.matrix = 0

    def fit(self, data):
        """
        :param data: DDF
        :return: trained model
        """

        df, nfrag, tmp = self._ddf_inital_setup(data)

        n_components = self.settings.get('n_components')
        cols = self.settings.get('input_col')

        partial_count = [pca_count(df[f], cols) for f in range(nfrag)]
        merged_count = merge_reduce(pca_merge_count, partial_count)

        for f in range(nfrag):
            partial_count[f] = partial_multiply(df[f], cols, merged_count)

        merged_cov = merge_reduce(pca_cov_merger, partial_count)
        info = pca_eigen_decomposition(merged_cov, n_components)
        compss_delete_object(partial_count)
        compss_delete_object(merged_count)
        compss_delete_object(merged_cov)

        self.var_exp, self.eig_vals, self.eig_vecs, self.matrix = info

        self.cum_var_exp = np.cumsum(self.var_exp)

        self.model = dict()
        self.model['algorithm'] = self.name
        # cumulative explained variance
        self.model['cum_var_exp'] = self.cum_var_exp
        self.model['eig_vals'] = self.eig_vals
        self.model['eig_vecs'] = self.eig_vecs
        self.model['model'] = self.matrix

        return self

    def fit_transform(self, data):
        """
        Fit the model and transform.

        :param data: DDF
        :return: DDF
        """

        self.fit(data)
        ddf = self.transform(data)

        return ddf

    def transform(self, data):
        """
        :param data: DDF
        :return: DDF
        """
        df, nfrag, tmp = self._ddf_inital_setup(data)

        if len(self.model) == 0:
            raise Exception("Model is not fitted.")

        model = self.model['model']
        features_col = self.settings['input_col']
        pred_col = self.settings['output_col']
        remove = self.settings['remove']

        result = [[] for _ in range(nfrag)]
        info = [[] for _ in range(nfrag)]
        for f in range(nfrag):
            result[f], info[f] = _pca_transform(df[f], features_col,
                                                pred_col, model, f, remove)

        uuid_key = self._ddf_add_task(task_name='transform_pca',
                                      status='COMPLETED', lazy=False,
                                      function={0: result},
                                      parent=[tmp.last_uuid],
                                      n_output=1, n_input=1, info=info)

        # a ml.transform will always have cache() before
        self._set_n_input(uuid_key, 0)
        return DDF(task_list=tmp.task_list, last_uuid=uuid_key)


@task(returns=1)
def pca_count(data, col):
    """Partial count."""
    partial_size = len(data)
    partial_sum = 0
    if partial_size > 0:
        partial_sum = data[col].values.sum(axis=0)
    return [partial_size, partial_sum]


@task(returns=1)
def pca_merge_count(count1, count2):
    """Merge partial counts."""
    partial_size = count1[0] + count2[0]
    partial_sum = np.add(count1[1], count2[1])
    return [partial_size, partial_sum]


@task(returns=1)
def partial_multiply(data, col, info):
    """Perform partial calculation."""
    cov_mat = 0
    total_size = info[0]

    if len(data) > 0:
        mean_vec = np.array(info[1]) / total_size

        x_std = data[col].values

        first_part = x_std - mean_vec
        cov_mat = first_part.T.dot(first_part)

    return [cov_mat, total_size]


@task(returns=1)
def pca_cov_merger(info1, info2):
    """Merge covariance."""
    cov1, total_size = info1
    cov2, _ = info2

    return [np.add(cov1, cov2), total_size]


# @local
def pca_eigen_decomposition(info, n_components):
    """Generate an eigen decomposition."""
    info = compss_wait_on(info)
    cov_mat, total_size = info
    dim = len(cov_mat)
    n_components = min([n_components, dim])
    cov_mat = cov_mat / (total_size-1)
    eig_vals, eig_vecs = np.linalg.eig(cov_mat)
    eig_vals = np.abs(eig_vals)

    total_values = sum(eig_vals)
    var_exp = [i*100/total_values for i in eig_vals]

    # Sort the eigenvalue and vecs tuples from high to low
    idxs = eig_vals.argsort()[::-1]
    eig_vals = eig_vals[idxs]
    eig_vecs = eig_vecs[idxs]

    matrix_w = eig_vecs[:, :n_components]

    return var_exp, eig_vals, eig_vecs, matrix_w


@task(returns=2)
def _pca_transform(data, features, pred_col, matrix_w, frag, remove):
    """Reduce the dimensionality based in the created model."""
    n_components = min([len(pred_col), len(matrix_w)])
    pred_col = pred_col[0: n_components]

    if len(data) > 0:
        array = data[features].values

        if not remove:
            to_remove = [c for c in pred_col if c in data.columns]
        else:
            to_remove = features

        data.drop(to_remove, axis=1, inplace=True)

        res = array.dot(matrix_w)

        data = pd.concat([data, pd.DataFrame(res, columns=pred_col)], axis=1)

    else:
        for col in pred_col:
            data[col] = np.nan

    info = generate_info(data, frag)
    return data, info

