#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Lucas Miguel S Ponce"
__email__ = "lucasmsp@gmail.com"

from pycompss.api.task import task
from pycompss.functions.reduce import merge_reduce
from pycompss.api.api import compss_wait_on
# from pycompss.api.local import *  # requires guppy

from ddf_library.ddf import DDF
from ddf_library.ddf_model import ModelDDF
from ddf_library.utils import generate_info, _feature_to_array

import numpy as np
import pandas as pd


class Binarizer(object):
    """
    Binarize data (set feature values to 0 or 1) according to a threshold

    Values greater than the threshold map to 1, while values less than or equal
    to the threshold map to 0. With the default threshold of 0, only positive
    values map to 1.

    :Example:

    >>> ddf = Binarizer(input_col='features', threshold=5.0).transform(ddf)
    """

    def __init__(self, input_col, output_col=None, threshold=0.0):
        """
        :param input_col: List of columns;
        :param output_col: Output column name.
        :param threshold: Feature values below or equal to this are
         replaced by 0, above it by 1. Default = 0.0.
        """

        if not output_col:
            output_col = 'features'

        self.settings = dict()
        self.settings['inputcol'] = input_col
        self.settings['outputcol'] = output_col
        self.settings['threshold'] = threshold

    def transform(self, data):
        """
        :param data: DDF
        :return: DDF
        """

        def task_binarizer(df, params):
            return _binarizer(df, params)

        uuid_key = data._ddf_add_task(task_name='binarizer',
                                      status='WAIT', lazy=True,
                                      function=[task_binarizer,
                                                self.settings],
                                      parent=[data.last_uuid],
                                      n_output=1, n_input=1)

        data._set_n_input(uuid_key, data.settings['input'])
        return DDF(task_list=data.task_list, last_uuid=uuid_key)


def _binarizer(df, settings):
    output_col = settings['outputcol']
    frag = settings['id_frag']

    if len(df) == 0:
        df[output_col] = np.nan

    else:
        input_col = settings['inputcol']
        threshold = settings['threshold']

        values = np.array(df[input_col].tolist())
        from sklearn.preprocessing import Binarizer
        values = Binarizer(threshold=threshold).fit_transform(values)
        df[output_col] = values.tolist()

    info = generate_info(df, frag)
    return df, info


class OneHotEncoder(ModelDDF):
    """
    Encode categorical integer features as a one-hot numeric array.

    :Example:

    >>> enc = OneHotEncoder(input_col='col_1', output_col='col_2')
    >>> ddf2 = enc.fit_transform(ddf1)
    """

    def __init__(self, input_col, output_col=None):
        """
        :param input_col: Input column name with the tokens;
        :param output_col: Output column name;
        """
        super(OneHotEncoder, self).__init__()

        if not isinstance(input_col, list):
            input_col = [input_col]

        if output_col is None:
            output_col = 'features_onehot'
        elif isinstance(output_col, list):
            raise Exception("You must insert only one name to output.")

        self.settings = dict()
        self.settings['input_col'] = input_col
        self.settings['output_col'] = output_col

        self.model = {}
        self.name = 'OneHotEncoder'

    def fit(self, data):
        """
        Fit the model.

        :param data: DDF
        :return: a trained model
        """

        df, nfrag, tmp = self._ddf_inital_setup(data)

        result_p = [[] for _ in range(nfrag)]
        for f in range(nfrag):
            result_p[f] = _one_hot_encoder(df[f], self.settings)

        categories = merge_reduce(_one_hot_encoder_merge, result_p)

        self.model['categories'] = compss_wait_on(categories)
        self.model['name'] = 'OneHotEncoder'

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
        if len(self.model) == 0:
            raise Exception("Model is not fitted.")

        df, nfrag, tmp = self._ddf_inital_setup(data)

        categories = self.model['categories']

        result = [[] for _ in range(nfrag)]
        info = [[] for _ in range(nfrag)]
        for f in range(nfrag):
            result[f], info[f] = _transform_one_hot(df[f], categories,
                                                    self.settings, f)

        uuid_key = self._ddf_add_task(task_name='transform_one_hot_encoder',
                                      status='COMPLETED', lazy=False,
                                      function={0: result},
                                      parent=[tmp.last_uuid],
                                      n_output=1, n_input=1, info=info)

        self._set_n_input(uuid_key, 0)
        return DDF(task_list=tmp.task_list, last_uuid=uuid_key)


@task(returns=1)
def _one_hot_encoder(df, settings):
    from sklearn.preprocessing import OneHotEncoder
    input_col = settings['input_col']
    X = df[input_col].values

    if len(X) > 0:
        enc = OneHotEncoder(handle_unknown='ignore')
        enc.fit(X)
        categories = enc.categories_
    else:
        categories = [np.array([]) for _ in range(len(input_col))]

    return categories


@task(returns=1)
def _one_hot_encoder_merge(x1, x2):

    x = []
    for i1, i2 in zip(x1, x2):
        t = np.unique(np.concatenate((i1, i2), axis=0))
        x.append(t)

    return x


@task(returns=2)
def _transform_one_hot(df, categories, settings, frag):
    from sklearn.preprocessing import OneHotEncoder
    input_col = settings['input_col']
    output_col = settings['output_col']

    if len(df) == 0:
        df[output_col] = np.nan
    else:
        enc = OneHotEncoder(handle_unknown='ignore', sparse=False, dtype=np.int)
        enc._legacy_mode = False
        enc.categories_ = categories

        df[output_col] = enc.transform(df[input_col].values).tolist()

    info = generate_info(df, frag)
    return df, info


class PolynomialExpansion(object):

    """
    Perform feature expansion in a polynomial space. As said in wikipedia of
    Polynomial Expansion, “In mathematics, an expansion of a product of sums
    expresses it as a sum of products by using the fact that multiplication
    distributes over addition”.

    For example, if an input sample is two dimensional and of the form [a, b],
    the degree-2 polynomial features are [1, a, b, a^2, ab, b^2]
    """

    def __init__(self, input_col, degree=2, output_col=None,
                 interaction_only=False):
        """
       :param input_col: List of columns;
       :param output_col: Output column name.
       :param degree: The degree of the polynomial features. Default = 2.
       :param interaction_only: If true, only interaction features are
        produced: features that are products of at most degree distinct input
        features. Default = False
       """

        if not output_col:
            output_col = 'features'

        self.settings = dict()
        self.settings['inputcol'] = input_col
        self.settings['outputcol'] = output_col
        self.settings['degree'] = int(degree)
        self.settings['interaction_only'] = interaction_only

    def transform(self, data):
        """
        :param data: DDF
        :return: DDF
        """

        def task_poly_expansion(df, params):
            return _poly_expansion(df, params)

        uuid_key = data._ddf_add_task(task_name='poly_expansion',
                                      status='WAIT', lazy=True,
                                      function=[task_poly_expansion,
                                                self.settings],
                                      parent=[data.last_uuid],
                                      n_output=1, n_input=1)

        data._set_n_input(uuid_key, data.settings['input'])
        return DDF(task_list=data.task_list, last_uuid=uuid_key)


def _poly_expansion(df, params):

    output_col = params['outputcol']
    frag = params['id_frag']

    if len(df) == 0:
        df[output_col] = np.nan
    else:
        input_col = params['inputcol']
        interaction_only = params['interaction_only']
        degree = params['degree']

        from sklearn.preprocessing import PolynomialFeatures
        values = np.array(df[input_col].tolist())
        values = PolynomialFeatures(degree=degree,
                                    interaction_only=interaction_only)\
            .fit_transform(values)
        df[output_col] = values.tolist()

    info = generate_info(df, frag)
    return df, info


class StringIndexer(ModelDDF):
    """
    StringIndexer indexes a feature by encoding a string column as a
    column containing indexes.
    :Example:
    >>> model = StringIndexer(input_col='category').fit(ddf1)
    >>> ddf2 = model.transform(ddf1)
    """

    def __init__(self, input_col, output_col=None):
        """
        :param input_col: Input string column;
        :param output_col:  Output indexes column.
        """
        super(StringIndexer, self).__init__()

        if not output_col:
            output_col = "{}_indexed".format(input_col)

        self.settings = dict()
        self.settings['input_col'] = input_col
        self.settings['output_col'] = output_col

        self.model = {}
        self.name = 'StringIndexer'

    def fit(self, data):
        """
        Fit the model.
        :param data: DDF
        :return: a trained model
        """

        in_col = self.settings['input_col']

        df, nfrag, tmp = self._ddf_inital_setup(data)

        mapper = [get_indexes(df[f], in_col) for f in range(nfrag)]
        mapper = merge_reduce(merge_mapper, mapper)

        self.model['model'] = compss_wait_on(mapper)
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

        if len(self.model) == 0:
            raise Exception("Model is not fitted.")

        in_col = self.settings['input_col']
        out_col = self.settings['output_col']

        df, nfrag, tmp = self._ddf_inital_setup(data)

        result = [[] for _ in range(nfrag)]
        info = [[] for _ in range(nfrag)]
        for f in range(nfrag):
            result[f], info[f] = _string_to_indexer(df[f], in_col, out_col,
                                                    self.model['model'], f)

        uuid_key = self._ddf_add_task(task_name='transform_string_indexer',
                                      status='COMPLETED', lazy=False,
                                      function={0: result},
                                      parent=[tmp.last_uuid],
                                      n_output=1, n_input=1, info=info)

        self._set_n_input(uuid_key, 0)
        return DDF(task_list=tmp.task_list, last_uuid=uuid_key)


@task(returns=1)
def get_indexes(data, in_col):
    """Create partial model to convert string to index."""
    return data[in_col].dropna().unique()


@task(returns=1)
def merge_mapper(data1, data2):
    """Merge partial models into one."""
    data1 = np.concatenate((data1, data2), axis=0)
    return np.unique(data1)


@task(returns=2)
def _string_to_indexer(data, in_col, out_col, mapper, frag):
    """Convert string to index based in the model."""
    news = [i for i in range(len(mapper))]
    mapper = mapper.tolist()
    data[out_col] = data[in_col].replace(to_replace=mapper, value=news)

    info = generate_info(data, frag)
    return data, info


class IndexToString(ModelDDF):
    """
    Symmetrically to StringIndexer, IndexToString maps a column of
    label indices back to a column containing the original labels as strings.
    :Example:
    >>> ddf2 = IndexToString(input_col='category_indexed',
    >>>                      model=model).transform(ddf1)
    """

    def __init__(self, input_col, model, output_col=None):
        """
        :param input_col: Input column name;
        :param model: Model generated by StringIndexer;
        :param output_col: Output column name.
        """
        super(IndexToString, self).__init__()

        if not output_col:
            output_col = "{}_converted".format(input_col)

        self.settings = dict()
        self.settings['input_col'] = input_col
        self.settings['output_col'] = output_col

        self.model = model.model
        self.name = 'IndexToString'

    def transform(self, data):

        if len(self.model) == 0:
            raise Exception("Model is not fitted.")

        input_col = self.settings['input_col']
        output_col = self.settings['output_col']

        df, nfrag, tmp = self._ddf_inital_setup(data)

        result = [[] for _ in range(nfrag)]
        info = [[] for _ in range(nfrag)]
        for f in range(nfrag):
            result[f], info[f] = _index_to_string(df[f], input_col,
                                                  output_col,
                                                  self.model['model'], f)

        uuid_key = self._ddf_add_task(task_name='transform_indextostring',
                                      status='COMPLETED', lazy=False,
                                      function={0: result},
                                      parent=[tmp.last_uuid],
                                      n_output=1, n_input=1, info=info)

        self._set_n_input(uuid_key, 0)
        return DDF(task_list=tmp.task_list, last_uuid=uuid_key)


@task(returns=2)
def _index_to_string(data, input_col, output_col, mapper, frag):
    """Convert index to string based in the model."""
    news = [i for i in range(len(mapper))]
    mapper = mapper.tolist()
    data[output_col] = data[input_col].replace(to_replace=news, value=mapper)

    info = generate_info(data, frag)
    return data, info
