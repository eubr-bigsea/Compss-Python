#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Lucas Miguel S Ponce"
__email__ = "lucasmsp@gmail.com"

from ddf_library.bases.context_base import ContextBase
from pycompss.api.task import task
from pycompss.functions.reduce import merge_reduce
from pycompss.api.api import compss_wait_on
from pycompss.api.parameter import FILE_IN

from ddf_library.bases.metadata import OPTGroup
from ddf_library.ddf import DDF
from ddf_library.bases.ddf_model import ModelDDF
from ddf_library.utils import generate_info, read_stage_file

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

    >>> ddf = Binarizer(threshold=5.0).transform(ddf, input_col=['feature'])
    """

    def __init__(self, threshold=0.0):
        """
        :param threshold: Feature values below or equal to this are
         replaced by 0, above it by 1. Default = 0.0;
        """

        super(Binarizer, self).__init__()
        self.threshold = threshold

    def transform(self, data, input_col, output_col=None, remove=False):
        """
        :param data: DDF
        :param input_col: List of columns;
        :param output_col: Output columns names.
        :param remove: Remove input columns after execution (default, False).
        :return: DDF
        """

        self.remove = remove

        if not isinstance(input_col, list):
            input_col = [input_col]
        self.input_col = input_col

        if output_col is None:
            output_col = '_binarized'
        if not isinstance(output_col, list):
            output_col = ['{}{}'.format(col, output_col) for col in input_col]
        self.output_col = output_col

        self.settings = self.__dict__.copy()

        uuid_key = ContextBase \
            .ddf_add_task(operation=self, parent=[data.last_uuid])

        return DDF(last_uuid=uuid_key)

    @staticmethod
    def function(df, params):
        return _binarizer(df, params)


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

    >>> enc = OneHotEncoder()
    >>> ddf2 = enc.fit_transform(ddf1, input_col='col_1', output_col='col_2')
    """

    def __init__(self):
        """
        """
        super(OneHotEncoder, self).__init__()

    def fit(self, data, input_col):
        """
        Fit the model.

        :param data: DDF
        :param input_col: Input column name with the tokens;
        :return: a trained model
        """

        df, nfrag, tmp = self._ddf_initial_setup(data)

        if not isinstance(input_col, list):
            input_col = [input_col]
        self.input_col = input_col

        result_p = [[] for _ in range(nfrag)]
        for f in range(nfrag):
            result_p[f] = _one_hot_encoder(df[f], self.input_col)

        categories = merge_reduce(_one_hot_encoder_merge, result_p)
        categories = compss_wait_on(categories)

        self.model['model'] = categories
        self.model['algorithm'] = self.name

        return self

    def fit_transform(self, data, input_col, output_col='_onehot',
                      remove=False):
        """
        Fit the model and transform.

        :param data: DDF
        :param input_col: Input column name with the tokens;
        :param output_col: Output suffix name. The pattern will be
         `col + order + suffix`, suffix default is '_onehot';
        :param remove: Remove input columns after execution (default, False).
        :return: DDF
        """

        self.fit(data, input_col)
        ddf = self.transform(data, output_col=output_col, remove=remove)

        return ddf

    def transform(self, data, input_col=None, output_col='_onehot',
                  remove=False):
        """
        :type output_col: str

        :param data: DDF
        :param input_col: Input column name with the tokens;
        :param output_col: Output suffix name. The pattern will be
         `col + order + suffix`, suffix default is '_onehot';
        :param remove: Remove input columns after execution (default, False).
        :return: DDF
        """

        self.check_fitted_model()
        if input_col:
            self.input_col = input_col

        dimension = sum([len(cat) for cat in self.model['model']])
        output_col = ['col{}{}'.format(i, output_col) for i in range(dimension)]
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
        return _transform_one_hot(df, params)


@task(returns=1, data_input=FILE_IN)
def _one_hot_encoder(data_input, input_col):
    from sklearn.preprocessing import OneHotEncoder

    df = read_stage_file(data_input)
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
        enc.drop_idx_ = None

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

    def __init__(self,  degree=2, interaction_only=False):
        # noinspection PyUnresolvedReferences
        """
       :param degree: The degree of the polynomial features. Default = 2.
       :param interaction_only: If true, only interaction features are
        produced: features that are products of at most degree distinct input
        features. Default = False

        :Example:

        >>> dff = PolynomialExpansion(degree=2)\
        >>>         .transform(ddf, input_col=['x', 'y'],)
       """
        super(PolynomialExpansion, self).__init__()

        self.degree = int(degree)
        self.interaction_only = interaction_only

    def transform(self, data, input_col, output_col="_poly",
                  remove=False):
        """
        :param data: DDF
        :param input_col: List of columns;
        :param output_col: Output suffix name following `col` plus order and
         suffix. Suffix default is '_poly';
        :param remove: Remove input columns after execution (default, False).
        :return: DDF
        """

        if not isinstance(input_col, list):
            input_col = [input_col]
        self.input_col = input_col

        if isinstance(output_col, list):
            raise Exception("'output_col' must be a single suffix name")
        self.output_col = output_col
        self.remove = remove

        self.settings = self.__dict__.copy()
        self.settings = _check_dimension(self.settings)

        uuid_key = ContextBase \
            .ddf_add_task(operation=self, parent=[data.last_uuid])

        return DDF(last_uuid=uuid_key)

    @staticmethod
    def function(df, params):
        params = params.copy()
        return _poly_expansion(df, params)


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
    >>> model = StringIndexer().fit(ddf1, input_col='category')
    >>> ddf2 = model.transform(ddf1)
    """

    def __init__(self):
        super(StringIndexer, self).__init__()

    def fit(self, data, input_col):
        """
        Fit the model.
        :param data: DDF
        :param input_col: list of columns;
        :return: a trained model
        """

        self.input_col = [input_col] \
            if isinstance(input_col, str) else input_col

        df, nfrag, tmp = self._ddf_initial_setup(data)

        mapper = [get_indexes(df[f], self.input_col) for f in range(nfrag)]
        mapper = merge_reduce(merge_mapper, mapper)

        self.model['model'] = compss_wait_on(mapper)
        self.model['algorithm'] = self.name

        return self

    def fit_transform(self, data, input_col, output_col=None):
        """
        Fit the model and transform.
        :param data: DDF
        :param input_col: list of columns;
        :param output_col:  Output indexes column.
        :return: DDF
        """

        self.fit(data, input_col)
        ddf = self.transform(data, input_col, output_col=output_col)

        return ddf

    def transform(self, data, input_col=None, output_col=None):
        """
        :param data: DDF
        :param input_col: list of columns;
        :param output_col:  List of output indexes column.
        :return: DDF
        """

        self.check_fitted_model()
        if input_col:
            self.input_col = input_col

        if output_col is None:
            output_col = ["{}_indexed".format(col)
                          for col in self.input_col]

        self.output_col = [output_col] \
            if isinstance(output_col, str) else output_col

        self.settings = self.__dict__.copy()

        uuid_key = ContextBase\
            .ddf_add_task(operation=self,
                          parent=[data.last_uuid])

        return DDF(last_uuid=uuid_key)

    @staticmethod
    def function(df, params):
        params['model'] = params['model']['model']
        return _string_to_indexer(df, params)


@task(returns=1, data_input=FILE_IN)
def get_indexes(data_input, in_cols):
    """Create partial model to convert string to index."""
    result = []
    data = read_stage_file(data_input, in_cols)
    for in_col in in_cols:
        result.extend(data[in_col].dropna().unique().tolist())
    result = np.unique(result).tolist()
    return result


@task(returns=1)
def merge_mapper(data1, data2):
    """Merge partial models into one."""
    data1 += data2
    del data2
    data1 = np.unique(data1).tolist()
    return data1


def _string_to_indexer(data, settings):
    """Convert string to index based in the model."""
    in_cols = settings['input_col']
    out_cols = settings['output_col']
    mapper = settings['model']
    frag = settings['id_frag']

    for in_col, out_col in zip(in_cols, out_cols):
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
    >>> ddf2 = IndexToString(model=model)\
    >>>         .transform(ddf1, input_col='category_indexed')
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

        self.input_col = [input_col] \
            if isinstance(input_col, str) else input_col

        if not output_col:
            self.output_col = ["{}_converted".format(col)
                               for col in self.input_col]

        self.settings = self.__dict__.copy()

        uuid_key = ContextBase \
            .ddf_add_task(operation=self, parent=[data.last_uuid])

        return DDF(last_uuid=uuid_key)

    @staticmethod
    def function(df, params):
        params = params.copy()
        params['model'] = params['model']['model']
        return _index_to_string(df, params)


def _index_to_string(data, settings):
    """Convert index to string based in the model."""
    input_col = settings['input_col']
    output_col = settings['output_col']
    mapper = settings['model']
    frag = settings['id_frag']
    for in_col, out_col in zip(input_col, output_col):
        news = np.arange(len(mapper), dtype=int)
        data[out_col] = data[in_col].replace(to_replace=news, value=mapper)

    info = generate_info(data, frag)
    return data, info
