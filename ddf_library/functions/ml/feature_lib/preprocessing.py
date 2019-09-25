#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Lucas Miguel S Ponce"
__email__ = "lucasmsp@gmail.com"

from pycompss.api.task import task
from pycompss.functions.reduce import merge_reduce
from pycompss.api.api import compss_wait_on

from ddf_library.ddf import DDF
from ddf_library.ddf_model import ModelDDF
from ddf_library.utils import generate_info

import numpy as np
import pandas as pd


class Binarizer(ModelDDF):
    # noinspection PyUnresolvedReferences
    """
    Binarize data (set feature values to 0 or 1) according to a threshold

    Values greater than the threshold map to 1, while values less than or equal
    to the threshold map to 0. With the default threshold of 0, only positive
    values map to 1.

    :Example:

    >>> ddf = Binarizer(input_col=['feature'], threshold=5.0).transform(ddf)
    """

    def __init__(self, input_col, threshold=0.0, remove=False):
        """
        :param input_col: List of columns;
        :param threshold: Feature values below or equal to this are
         replaced by 0, above it by 1. Default = 0.0;
        :param remove: Remove input columns after execution (default, False).
        """

        super(Binarizer, self).__init__()

        if not isinstance(input_col, list):
            input_col = [input_col]

        self.settings = dict()
        self.settings['input_col'] = input_col
        self.settings['threshold'] = threshold
        self.settings['remove'] = remove
        self.output_cols = None

    def transform(self, data, output_col=None):
        """
        :param data: DDF
        :param output_col: Output columns names.
        :return: DDF
        """

        input_col = self.settings['input_col']

        if output_col is None:
            output_col = '_binarized'

        if not isinstance(output_col, list):
            output_col = ['{}{}'.format(col, output_col) for col in
                          input_col]

        self.settings['output_col'] = output_col
        self.output_cols = output_col

        def task_binarizer(df, params):
            return _binarizer(df, params)

        uuid_key = self._ddf_add_task(task_name='binarizer',
                                      opt=self.OPT_SERIAL,
                                      function=[task_binarizer, self.settings],
                                      parent=[data.last_uuid])

        return DDF(task_list=data.task_list, last_uuid=uuid_key)


def _binarizer(df, settings):

    frag = settings['id_frag']
    input_col = settings['input_col']
    output_col = settings['output_col']
    threshold = settings['threshold']
    remove_input = settings.get('remove', False)

    values = df[input_col].values
    to_remove = [c for c in output_col if c in df.columns]
    if remove_input:
        to_remove += input_col
    df.drop(to_remove, axis=1, inplace=True)

    if len(df) > 0:

        from sklearn.preprocessing import Binarizer
        values = Binarizer(threshold=threshold).fit_transform(values)

        df = pd.concat([df, pd.DataFrame(values, columns=output_col)], axis=1)

    else:
        for col in output_col:
            df[col] = np.nan

    info = generate_info(df, frag)
    return df, info


class OneHotEncoder(ModelDDF):
    # noinspection PyUnresolvedReferences
    """
    Encode categorical integer features as a one-hot numeric array.

    :Example:

    >>> enc = OneHotEncoder(input_col='col_1', output_col='col_2')
    >>> ddf2 = enc.fit_transform(ddf1)
    """

    def __init__(self, input_col, remove=False):
        """
        :param input_col: Input column name with the tokens;
        :param remove: Remove input columns after execution (default, False).
        """
        super(OneHotEncoder, self).__init__()

        if not isinstance(input_col, list):
            input_col = [input_col]

        self.settings = dict()
        self.settings['input_col'] = input_col
        self.settings['remove'] = remove

        self.model = {}
        self.name = 'OneHotEncoder'
        self.output_cols = None

    def fit(self, data):
        """
        Fit the model.

        :param data: DDF
        :return: a trained model
        """

        df, nfrag, tmp = self._ddf_initial_setup(data)

        result_p = [[] for _ in range(nfrag)]
        for f in range(nfrag):
            result_p[f] = _one_hot_encoder(df[f], self.settings)

        categories = merge_reduce(_one_hot_encoder_merge, result_p)
        categories = compss_wait_on(categories)

        self.model['model'] = categories
        self.model['algorithm'] = self.name

        return self

    def fit_transform(self, data, output_col='_onehot'):
        """
        Fit the model and transform.

        :param data: DDF
        :param output_col: Output suffix name. The pattern will be
         `col + order + suffix`, suffix default is '_onehot';
        :return: DDF
        """

        self.fit(data)
        ddf = self.transform(data, output_col)

        return ddf

    def transform(self, data, output_col='_onehot'):
        """
        :type output_col: str

        :param data: DDF
        :param output_col: Output suffix name. The pattern will be
         `col + order + suffix`, suffix default is '_onehot';
        :return: DDF
        """

        self.check_fitted_model()

        task_list = data.task_list
        settings = self.settings.copy()
        settings['model'] = self.model['model'].copy()

        dimension = sum([len(cat) for cat in settings['model']])
        output_col = ['col{}{}'.format(i, output_col) for i in range(dimension)]
        settings['output_col'] = output_col

        def task_transform_one_hot(df, params):
            return _transform_one_hot(df, params)

        uuid_key = self._ddf_add_task(task_name='task_transform_one_hot',
                                      opt=self.OPT_SERIAL,
                                      function=[task_transform_one_hot,
                                                settings],
                                      parent=[data.last_uuid])

        return DDF(task_list=task_list, last_uuid=uuid_key)


@task(returns=1)
def _one_hot_encoder(df, settings):
    from sklearn.preprocessing import OneHotEncoder
    input_col = settings['input_col']
    values = df[input_col].values
    del df

    if len(values) > 0:
        enc = OneHotEncoder(handle_unknown='ignore')
        enc.fit(values)
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


def _transform_one_hot(df, settings):
    from sklearn.preprocessing import OneHotEncoder
    categories = settings['model']
    frag = settings['id_frag']
    input_col = settings['input_col']
    output_col = settings['output_col']
    remove_input = settings.get('remove', False)

    values = df[input_col].values
    to_remove = [c for c in output_col if c in df.columns]
    if remove_input:
        to_remove += input_col
    df.drop(to_remove, axis=1, inplace=True)

    if len(df) > 0:
        enc = OneHotEncoder(handle_unknown='ignore', sparse=False, dtype=np.int)
        enc._legacy_mode = False
        enc.categories_ = categories

        values = enc.transform(values).tolist()

        df = pd.concat([df, pd.DataFrame(values, columns=output_col)], axis=1)

    else:
        for col in output_col:
            df[col] = np.nan

    info = generate_info(df, frag)
    return df, info


class PolynomialExpansion(ModelDDF):

    """
    Perform feature expansion in a polynomial space. In mathematics,
    an expansion of a product of sums expresses it as a sum of products by
    using the fact that multiplication distributes over addition.

    For example, if an input sample is two dimensional and of the form [a, b],
    the degree-2 polynomial features are [1, a, b, a^2, ab, b^2]
    """

    def __init__(self, input_col, degree=2, interaction_only=False,
                 remove=False):
        # noinspection PyUnresolvedReferences
        """
       :param input_col: List of columns;
       :param degree: The degree of the polynomial features. Default = 2.
       :param interaction_only: If true, only interaction features are
        produced: features that are products of at most degree distinct input
        features. Default = False
       :param remove: Remove input columns after execution (default, False).

        :Example:

        >>> dff = PolynomialExpansion(input_col=['x', 'y'],
        >>>                           degree=2).transform(ddf)
       """
        super(PolynomialExpansion, self).__init__()

        if not isinstance(input_col, list):
            input_col = [input_col]

        self.settings = dict()
        self.settings['input_col'] = input_col
        self.settings['degree'] = int(degree)
        self.settings['interaction_only'] = interaction_only
        self.settings['remove'] = remove
        self.output_cols = None

    def transform(self, data, output_col="_poly"):
        """
        :param data: DDF
        :param output_col: Output suffix name following `col` plus order and
         suffix. Suffix default is '_poly';
        :return: DDF
        """

        if isinstance(output_col, list):
            raise Exception("'output_col' must be a single suffix name")

        settings = self.settings.copy()
        # noinspection PyTypeChecker
        settings['output_col'] = output_col
        settings = _check_dimension(settings)

        def task_poly_expansion(df, params):
            return _poly_expansion(df, params)

        uuid_key = self._ddf_add_task(task_name='poly_expansion',
                                      opt=self.OPT_SERIAL,
                                      function=[task_poly_expansion, settings],
                                      parent=[data.last_uuid])

        return DDF(task_list=data.task_list, last_uuid=uuid_key)


def _check_dimension(params):
    """Create a simple data to check the new dimension."""
    input_col = params['input_col']
    output_col = params['output_col']

    interaction_only = params['interaction_only']
    degree = params['degree']

    from sklearn.preprocessing import PolynomialFeatures
    poly = PolynomialFeatures(degree=degree, interaction_only=interaction_only)
    tmp = [[0 for _ in range(len(input_col))]]
    dim = poly.fit_transform(tmp).shape[1]

    params['output_col'] = ['col{}{}'.format(i, output_col) for i in range(dim)]
    return params


def _poly_expansion(df, params):
    input_col = params['input_col']
    output_col = params['output_col']
    frag = params['id_frag']
    remove_input = params.get('remove', False)
    interaction_only = params['interaction_only']
    degree = params['degree']

    values = df[input_col].values
    to_remove = [c for c in output_col if c in df.columns]
    if remove_input:
        to_remove += input_col
    df.drop(to_remove, axis=1, inplace=True)

    if len(df) > 0:
        from sklearn.preprocessing import PolynomialFeatures
        poly = PolynomialFeatures(degree=degree,
                                  interaction_only=interaction_only)

        values = poly.fit_transform(values)
        df = pd.concat([df, pd.DataFrame(values, columns=output_col)], axis=1)

    else:
        for col in output_col:
            df[col] = np.nan

    info = generate_info(df, frag)
    return df, info


class StringIndexer(ModelDDF):
    # noinspection PyUnresolvedReferences
    """
    StringIndexer indexes a feature by encoding a string column as a
    column containing indexes.
    :Example:
    >>> model = StringIndexer(input_col='category').fit(ddf1)
    >>> ddf2 = model.transform(ddf1)
    """

    def __init__(self):
        super(StringIndexer, self).__init__()

        self.settings = dict()

        self.model = {}
        self.name = 'StringIndexer'

    def fit(self, data, input_col):
        """
        Fit the model.
        :param data: DDF
        :param input_col: Input string column;
        :return: a trained model
        """

        self.settings['input_col'] = input_col

        df, nfrag, tmp = self._ddf_initial_setup(data)

        mapper = [get_indexes(df[f], input_col) for f in range(nfrag)]
        mapper = merge_reduce(merge_mapper, mapper)

        self.model['model'] = compss_wait_on(mapper)
        self.model['algorithm'] = self.name
        return self

    def fit_transform(self, data, input_col, output_col=None):
        """
        Fit the model and transform.
        :param data: DDF
        :param input_col: Input string column;
        :param output_col:  Output indexes column.
        :return: DDF
        """

        self.fit(data, input_col)
        ddf = self.transform(data, output_col)

        return ddf

    def transform(self, data, output_col=None):
        """
        :param data: DDF
        :param output_col:  Output indexes column.
        :return: DDF
        """

        self.check_fitted_model()

        task_list = data.task_list
        settings = self.settings.copy()
        settings['model'] = self.model['model'].copy()

        if output_col is None:
            output_col = "{}_indexed".format(settings['input_col'])
        settings['output_col'] = output_col

        def task_string_to_indexer(df, params):
            return _string_to_indexer(df, params)

        uuid_key = self._ddf_add_task(task_name='task_string_to_indexer',
                                      opt=self.OPT_SERIAL,
                                      function=[task_string_to_indexer,
                                                settings],
                                      parent=[data.last_uuid])

        return DDF(task_list=task_list, last_uuid=uuid_key)


@task(returns=1)
def get_indexes(data, in_col):
    """Create partial model to convert string to index."""
    data = data[in_col].dropna().unique().tolist()
    return data


@task(returns=1)
def merge_mapper(data1, data2):
    """Merge partial models into one."""
    data1 += data2
    del data2
    data1 = np.unique(data1).tolist()
    return data1


def _string_to_indexer(data, settings):
    """Convert string to index based in the model."""
    in_col = settings['input_col']
    out_col = settings['output_col']
    mapper = settings['model']
    frag = settings['id_frag']

    news = np.arange(len(mapper), dtype=int)
    data[out_col] = data[in_col].replace(to_replace=mapper, value=news)

    info = generate_info(data, frag)
    return data, info


class IndexToString(ModelDDF):
    # noinspection PyUnresolvedReferences
    """
    Symmetrically to StringIndexer, IndexToString maps a column of
    label indices back to a column containing the original labels as strings.
    :Example:
    >>> ddf2 = IndexToString(input_col='category_indexed',
    >>>                      model=model).transform(ddf1)
    """

    def __init__(self, model):
        """
        :param model: Model generated by StringIndexer;
        """
        super(IndexToString, self).__init__()

        self.settings = dict()
        self.model = model.model
        self.name = model.name

    def transform(self, data, input_col, output_col=None):
        """

        :param data:
        :param input_col: Input column name;
        :param output_col: Output column name.
        :return:
        """

        self.check_fitted_model()
        self.settings['input_col'] = input_col

        task_list = data.task_list
        settings = self.settings.copy()
        settings['model'] = self.model['model'].copy()

        if not output_col:
            settings['output_col'] = str(input_col)+"_converted"

        def task_index_to_string(df, params):
            return _index_to_string(df, params)

        uuid_key = self._ddf_add_task(task_name='task_index_to_string',
                                      opt=self.OPT_SERIAL,
                                      function=[task_index_to_string, settings],
                                      parent=[data.last_uuid])

        return DDF(task_list=task_list, last_uuid=uuid_key)


def _index_to_string(data, settings):
    """Convert index to string based in the model."""
    input_col = settings['input_col']
    output_col = settings['output_col']
    mapper = settings['model']
    frag = settings['id_frag']
    news = np.arange(len(mapper), dtype=int)
    data[output_col] = data[input_col].replace(to_replace=news, value=mapper)

    info = generate_info(data, frag)
    return data, info
