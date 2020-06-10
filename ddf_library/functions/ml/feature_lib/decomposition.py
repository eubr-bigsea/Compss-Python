#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Lucas Miguel S Ponce"
__email__ = "lucasmsp@gmail.com"

from ddf_library.bases.metadata import OPTGroup
from ddf_library.bases.context_base import ContextBase
from pycompss.api.parameter import FILE_IN
from pycompss.api.task import task
from pycompss.functions.reduce import merge_reduce
from pycompss.api.api import compss_wait_on, compss_delete_object

from ddf_library.ddf import DDF
from ddf_library.bases.ddf_model import ModelDDF
from ddf_library.utils import generate_info, read_stage_file

import numpy as np
import pandas as pd


class PCA(ModelDDF):
    # noinspection PyUnresolvedReferences
    """
    Principal component analysis (PCA) is a statistical method to find
    a rotation such that the first coordinate has the largest variance
    possible, and each succeeding coordinate in turn has the largest
    variance possible. The columns of the rotation matrix are called
    principal components. PCA is used widely in dimensionality reduction.

    :Example:

    >>> pca = PCA(n_components=2).fit(ddf1, input_col='features')
    >>> ddf2 = pca.transform(ddf1, output_col='features_pca')
    """

    def __init__(self, n_components):
        """
        :param n_components: Number of output components;
        """
        super(PCA, self).__init__()

        self.n_components = n_components
        self.var_exp = self.cum_var_exp = \
            self.eig_values = self.eig_vectors = self.matrix = 0

    def fit(self, data, input_col):
        """
        :param data: DDF
        :param input_col: Input columns;
        :return: trained model
        """

        df, nfrag, tmp = self._ddf_initial_setup(data)

        if not isinstance(input_col, list):
            input_col = [input_col]
        self.input_col = input_col

        partial_count = [pca_count(df[f], input_col) for f in range(nfrag)]
        merged_count = merge_reduce(pca_merge_count, partial_count)

        for f in range(nfrag):
            partial_count[f] = partial_multiply(df[f], input_col, merged_count)

        merged_cov = merge_reduce(pca_cov_merger, partial_count)
        info = pca_eigen_decomposition(merged_cov, self.n_components)
        compss_delete_object(partial_count)
        compss_delete_object(merged_count)
        compss_delete_object(merged_cov)

        self.var_exp, self.eig_values, self.eig_vectors, self.matrix = info
        self.cum_var_exp = np.cumsum(self.var_exp)

        self.model = dict()
        self.model['algorithm'] = self.name
        # cumulative explained variance
        self.model['cum_var_exp'] = self.cum_var_exp
        self.model['eig_values'] = self.eig_values
        self.model['eig_vectors'] = self.eig_vectors
        self.model['model'] = self.matrix

        return self

    def fit_transform(self, data, input_col, output_col='_pca', remove=False):
        """
        Fit the model and transform.

        :param data: DDF
        :param input_col: Input columns;
        :param output_col: A list of output feature column or a suffix name.
        :param remove: Remove input columns after execution (default, False).
        :return: DDF
        """

        self.fit(data, input_col)
        ddf = self.transform(data, output_col=output_col, remove=remove)

        return ddf

    def transform(self, data, input_col=None, output_col='_pca', remove=False):
        """
        :param data: DDF
        :param input_col: Input columns;
        :param output_col: A list of output feature column or a suffix name.
        :param remove: Remove input columns after execution (default, False).
        :return: DDF
        """

        self.check_fitted_model()
        if not input_col:
            input_col = self.input_col

        if not isinstance(output_col, list):
            output_col = ['{}{}'.format(col, output_col) for col in input_col]
        self.output_col = output_col
        self.remove = remove

        self.settings = self.__dict__.copy()

        uuid_key = ContextBase \
            .ddf_add_task(operation=self, parent=[data.last_uuid])

        return DDF(last_uuid=uuid_key)

    @staticmethod
    def function(df, params):
        params = params.copy()
        params['model'] = params['model']['model']
        return _pca_transform(df, params)


@task(returns=1, data_input=FILE_IN)
def pca_count(data_input, col):
    """Partial count."""
    data = read_stage_file(data_input, col)
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


@task(returns=1, data_input=FILE_IN)
def partial_multiply(data_input, col, info):
    """Perform partial calculation."""
    cov_mat = 0
    total_size = info[0]
    data = read_stage_file(data_input, col)

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


def pca_eigen_decomposition(info, n_components):
    """Generate an eigen decomposition."""
    info = compss_wait_on(info)
    cov_mat, total_size = info
    dim = len(cov_mat)
    n_components = min([n_components, dim])
    cov_mat = cov_mat / (total_size-1)
    eig_values, eig_vectors = np.linalg.eig(cov_mat)
    eig_values = np.abs(eig_values)

    total_values = sum(eig_values)
    var_exp = [i*100/total_values for i in eig_values]

    # Sort the eigenvalue and vectors tuples from high to low
    idx = eig_values.argsort()[::-1]
    eig_values = eig_values[idx]
    eig_vectors = eig_vectors[idx]

    matrix_w = eig_vectors[:, :n_components]

    return var_exp, eig_values, eig_vectors, matrix_w


def _pca_transform(data, settings):
    """Reduce the dimensionality based in the created model."""
    features = settings['input_col']
    pred_col = settings['output_col']
    matrix_w = settings['model']
    frag = settings['id_frag']
    remove = settings['remove']

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
