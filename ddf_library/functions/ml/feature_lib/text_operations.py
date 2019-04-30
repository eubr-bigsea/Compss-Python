#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Lucas Miguel S Ponce"
__email__ = "lucasmsp@gmail.com"

from pycompss.api.task import task
from pycompss.functions.reduce import merge_reduce
from pycompss.api.api import compss_wait_on
# from pycompss.api.local import *  # requires guppy

from ddf_library.ddf import DDF, DDFSketch
from ddf_library.ddf_model import ModelDDF
from ddf_library.utils import generate_info

import numpy as np
import pandas as pd
import re
import itertools


class Tokenizer(object):
    """
    Tokenization is the process of taking text (such as a sentence) and
    breaking it into individual terms (usually words). A simple Tokenizer
    class provides this functionality.

    :Example:

    >>> ddf2 = Tokenizer(input_col='features').transform(ddf1)
    """

    def __init__(self, input_col, output_col=None, min_token_length=2,
                 to_lowercase=True):
        """
        :param input_col: Input column with sentences;
        :param output_col: Output column (*'input_col'_tokenized* if None);
        :param min_token_length: Minimum tokens length (default is 2);
        :param to_lowercase: To convert words to lowercase (default is True).
        """

        if isinstance(input_col, list):
            raise Exception('`input_col` must be a single column')

        if not output_col:
            output_col = "{}_tokenized".format(input_col)

        self.settings = dict()
        self.settings['inputcol'] = input_col
        self.settings['outputcol'] = output_col
        self.settings['min_token_length'] = min_token_length
        self.settings['to_lowercase'] = to_lowercase

    def transform(self, data):
        """

        :param data: DDF
        :return: DDF
        """

        def task_tokenizer(df, params):
            return _tokenizer_(df, params)

        uuid_key = data._ddf_add_task(task_name='tokenizer',
                                      status='WAIT', lazy=True,
                                      function=[task_tokenizer,
                                                self.settings],
                                      parent=[data.last_uuid],
                                      n_output=1, n_input=1)

        data._set_n_input(uuid_key, data.settings['input'])
        return DDF(task_list=data.task_list, last_uuid=uuid_key)


class RegexTokenizer(object):
    """
    A regex based tokenizer that extracts tokens either by using the provided
    regex pattern (in Java dialect) to split the text.

    :Example:

    >>> ddf2 = RegexTokenizer(input_col='col_0', pattern=r"(?u)\b\w\w+\b")\
    ...         .transform(ddf1)
    """

    def __init__(self, input_col, output_col=None, pattern=r'\s+',
                 min_token_length=2, to_lowercase=True):
        """
        :param input_col: Input column with sentences;
        :param output_col: Output column (*'input_col'_tokenized* if None);
        :param pattern: Regex pattern in Java dialect,
         default *r"(?u)\b\w\w+\b"*;
        :param min_token_length: Minimum tokens length (default is 2);
        :param to_lowercase: To convert words to lowercase (default is True).
        """

        if isinstance(input_col, list):
            raise Exception('`input_col` must be a single column')

        if not output_col:
            output_col = "{}_tokenized".format(input_col)

        self.settings = dict()
        self.settings['inputcol'] = input_col
        self.settings['outputcol'] = output_col
        self.settings['min_token_length'] = min_token_length
        self.settings['to_lowercase'] = to_lowercase
        self.settings['pattern'] = pattern

    def transform(self, data):
        """
        :param data: DDF
        :return: DDF
        """

        def task_regex_tokenizer(df, params):
            return _tokenizer_(df, params)

        uuid_key = data._ddf_add_task(task_name='tokenizer',
                                      status='WAIT', lazy=True,
                                      function=[task_regex_tokenizer,
                                                self.settings],
                                      parent=[data.last_uuid],
                                      n_output=1, n_input=1)

        data._set_n_input(uuid_key, data.settings['input'])
        return DDF(task_list=data.task_list, last_uuid=uuid_key)


def _tokenizer_(data, settings):
    """Perform a partial tokenizer."""

    input_col = settings['inputcol']
    output_col = settings['outputcol']
    min_token_length = settings['min_token_length']
    to_lowercase = settings['to_lowercase']
    pattern = settings.get('pattern', r"(?u)\b\w\w+\b")
    frag = settings['id_frag']

    token_pattern = re.compile(pattern)
    tokenizer = lambda doc: token_pattern.findall(doc)

    result = []
    if len(data) > 0:

        for sentence in data[input_col].values:
            toks = tokenizer(sentence)
            row = []
            for t in toks:
                if len(t) > min_token_length:
                    if to_lowercase:
                        row.append(t.lower())
                    else:
                        row.append(t)
            result.append(row)

    else:
        result = np.nan

    if output_col in data.columns:
        data.drop([output_col], axis=1, inplace=True)

    data[output_col] = result

    info = generate_info(data, frag)
    return data, info


class RemoveStopWords(DDFSketch):
    """
    Remove stop-words is a operation to remove words which
    should be excluded from the input, typically because
    the words appear frequently and don’t carry as much meaning.

    :Example:

    >>> remover = RemoveStopWords(input_col='col_0', output_col='col_1',
    >>>                           stops_words_list=['rock', 'destined'])
    >>> remover = remover.stopwords_from_ddf(stopswords_ddf, 'col')
    >>> ddf2 = remover.transform(ddf1)
    """

    def __init__(self, input_col, output_col=None, case_sensitive=True,
                 stops_words_list=None):
        """
        :param input_col: Input columns with the tokens;
        :param output_col: Output column (*'input_col'_rm_stopwords* if None);
        :param case_sensitive: To compare words using case sensitive (default);
        :param stops_words_list: Optional, a list of words to be removed.
        """
        super(RemoveStopWords, self).__init__()

        if isinstance(input_col, list):
            raise Exception('`input_col` must be a single column')

        if not output_col:
            output_col = "{}_rm_stopwords".format(input_col)

        self.settings = dict()
        self.settings['news_stops_words'] = stops_words_list
        self.settings['input_col'] = input_col
        self.settings['case_sensitive'] = case_sensitive
        self.settings['output_col'] = output_col

        self.name = 'RemoveStopWords'
        self.stopwords = []

    def stopwords_from_ddf(self, data, input_col):
        """
        Is also possible inform stop-words form a DDF.

        :param data: DDF with a column of stop-words;
        :param input_col: Stop-words column name;
        """

        # It assumes that stopwords's dataframe can fit in memmory
        df, nfrag, tmp = self._ddf_inital_setup(data)

        stopwords = [[] for _ in range(nfrag)]
        for f in range(nfrag):
            stopwords[f] = read_stopwords(df[f], input_col)

        stopwords = merge_reduce(merge_stopwords, stopwords)

        self.stopwords = stopwords
        return self

    def transform(self, data):
        """
        :param data: DDF
        :return: DDF
        """

        df, nfrag, tmp = self._ddf_inital_setup(data)

        result = [[] for _ in range(nfrag)]
        info = [[] for _ in range(nfrag)]
        for f in range(nfrag):
            result[f], info[f] = _remove_stopwords(df[f], self.settings,
                                                   self.stopwords, f)

        uuid_key = self._ddf_add_task(task_name='task_transform_stopwords',
                                      status='COMPLETED', lazy=False,
                                      function={0: result},
                                      parent=[tmp.last_uuid],
                                      n_output=1, n_input=1, info=info)

        self._set_n_input(uuid_key, 0)
        return DDF(task_list=tmp.task_list, last_uuid=uuid_key)


@task(returns=1)
def read_stopwords(data1, input_col):
    if len(data1) > 0:
        data1 = data1[input_col].values
    else:
        data1 = np.array([])
    return data1


@task(returns=1)
def merge_stopwords(data1, data2):

    data1 = np.concatenate((data1, data2), axis=0)
    return data1


@task(returns=2)
def _remove_stopwords(data, settings, stopwords, frag):
    """Remove stopwords from a column."""
    column = settings['input_col']
    output_col = settings['output_col']

    # stopwords must be in 1-D
    new_stops = np.reshape(settings['news_stops_words'], -1, order='C')
    if len(stopwords) != 0:
        stopwords = np.concatenate((stopwords, new_stops), axis=0)
    else:
        stopwords = new_stops

    tmp = []
    if data.shape[0] > 0:
        if settings['case_sensitive']:
            stopwords = set(stopwords)
            for tokens in data[column].values:
                tmp.append(list(set(tokens).difference(stopwords)))

        else:
            stopwords = set([tok.lower() for tok in stopwords])

            for tokens in data[column].values:
                entry = [tok.lower() for tok in tokens]
                tmp.append(list(set(entry).difference(stopwords)))

    else:
        tmp = np.nan

    if output_col in data.columns:
        data.drop([output_col], axis=1, inplace=True)

    data[output_col] = tmp

    info = generate_info(data, frag)
    return data, info


class NGram(object):
    """
    A feature transformer that converts the input array of strings into an
    array of n-grams. Null values in the input array are ignored. It returns
    an array of n-grams where each n-gram is represented by a space-separated
    string of words. When the input is empty, an empty array is returned. When
    the input array length is less than n (number of elements per n-gram), no
    n-grams are returned.
    """

    def __init__(self, input_col, n=2, output_col=None):
        """

        :param input_col: Input columns with the tokens;
        :param n: Number integer. Default = 2;
        :param output_col: Output column. Default, add suffix '_ngram';

        """
        if isinstance(input_col, list):
            raise Exception('`input_col` must be a single column')

        if not output_col:
            output_col = "{}_ngram".format(input_col)

        self.settings = dict()
        self.settings['inputcol'] = input_col
        self.settings['outputcol'] = output_col
        self.settings['n'] = n

    def transform(self, data):
        """
        :param data: DDF
        :return: DDF
        """

        def task_ngram(df, params):
            return _ngram(df, params)

        uuid_key = data._ddf_add_task(task_name='ngram',
                                      status='WAIT', lazy=True,
                                      function=[task_ngram, self.settings],
                                      parent=[data.last_uuid],
                                      n_output=1, n_input=1)

        data._set_n_input(uuid_key, data.settings['input'])
        return DDF(task_list=data.task_list, last_uuid=uuid_key)


def _ngram(df, settings):

    input_col = settings['inputcol']
    output_col = settings['outputcol']
    frag = settings['id_frag']

    if len(df) > 0:

        from nltk.util import ngrams

        n = settings['n']

        def ngrammer(row):
            return [" ".join(gram) for gram in ngrams(row, n)]

        tmp = [ngrammer(row) for row in df[input_col].values]

    else:
        tmp = np.nan

    if output_col in df.columns:
        df.drop([output_col], axis=1, inplace=True)

    df[output_col] = tmp

    info = generate_info(df, frag)
    return df, info




