#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Lucas Miguel S Ponce"
__email__ = "lucasmsp@gmail.com"

from pycompss.api.task import task
from pycompss.functions.reduce import merge_reduce
from pycompss.api.api import compss_wait_on
from ddf_library.ddf import DDF, DDFSketch
from ddf_library.ddf_model import ModelDDF
from ddf_library.utils import generate_info, _feature_to_array
# from pycompss.api.local import *  # requires guppy
import numpy as np
import pandas as pd
import itertools


class CountVectorizer(ModelDDF):
    """
    Converts a collection of text documents to a matrix of token counts.

    :Example:

    >>> cv = CountVectorizer(input_col='col_1', output_col='col_2').fit(ddf1)
    >>> ddf2 = cv.transform(ddf1)
    """

    def __init__(self, input_col, output_col=None, vocab_size=-1, min_tf=1.0,
                 min_df=1.0, binary=True):
        """
        :param input_col: Input column name with the tokens;
        :param output_col: Output column name;
        :param vocab_size: Maximum size of the vocabulary. If -1, no limits
         will be applied. (default, -1)
        :param min_tf: Specifies the minimum number of different documents a
         term must appear in to be included in the vocabulary. If this is an
         integer >= 1, this specifies the number of documents the term must
         appear in;  Default 1.0;
        :param min_df: Filter to ignore rare words in a document. For each
         document, terms with frequency/count less than the given threshold
         are ignored. If this is an integer >= 1, then this specifies a count
         (of times the term must appear in the document);
        :param binary: If True, all nonzero counts are set to 1.
        """
        super(CountVectorizer, self).__init__()

        if not output_col:
            output_col = input_col

        self.settings = dict()
        self.settings['input_col'] = [input_col]
        self.settings['output_col'] = output_col
        self.settings['vocab_size'] = vocab_size
        self.settings['min_tf'] = min_tf
        self.settings['min_df'] = min_df
        self.settings['binary'] = binary

        self.model = []
        self.name = 'CountVectorizer'

    def fit(self, data):
        """
        Fit the model.

        :param data: DDF
        :return: a trained model
        """

        vocab_size = self.settings['vocab_size']
        min_tf = self.settings['min_tf']
        min_df = self.settings['min_df']

        df, nfrag, tmp = self._ddf_inital_setup(data)

        result_p = [[] for _ in range(nfrag)]
        for f in range(nfrag):
            result_p[f] = wordCount(df[f], self.settings)
        word_dic = merge_reduce(merge_wordCount, result_p)

        vocabulary = create_vocabulary(word_dic)

        if any([min_tf > 0, min_df > 0, vocab_size > 0]):
            vocabulary = filter_words(vocabulary, self.settings)

        self.model = [vocabulary]

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

        vocabulary = self.model[0]

        df, nfrag, tmp = self._ddf_inital_setup(data)

        result = [[] for _ in range(nfrag)]
        info = [[] for _ in range(nfrag)]
        for f in range(nfrag):
            result[f], info[f] = _transform_BoW(df[f], vocabulary,
                                                self.settings)

        uuid_key = self._ddf_add_task(task_name='transform_count_vectorizer',
                                      status='COMPLETED', lazy=False,
                                      function={0: result},
                                      parent=[tmp.last_uuid],
                                      n_output=1, n_input=1, info=info)

        self._set_n_input(uuid_key, 0)
        return DDF(task_list=tmp.task_list, last_uuid=uuid_key)


@task(returns=dict)
def wordCount(data, params):
    """Auxilar method to create a model."""
    wordcount = {}
    columns = params['input_col']
    # first:   Number of all occorrences with term t
    # second:  Number of diferent documents with term t
    # third:   temporary - only to idetify the last occorrence

    for i_doc, doc in enumerate(data[columns].values):
        doc = np.array(list(itertools.chain(doc))).flatten()
        for token in doc:
            if token not in wordcount:
                wordcount[token] = [1, 1, i_doc]
            else:
                wordcount[token][0] += 1
                if wordcount[token][2] != i_doc:
                    wordcount[token][1] += 1
                    wordcount[token][2] = i_doc
    return wordcount


@task(returns=1)
def merge_wordCount(dic1, dic2):
    """Merge the wordcounts."""
    for k in dic2:
        if k in dic1:
            dic1[k][0] += dic2[k][0]
            dic1[k][1] += dic2[k][1]
        else:
            dic1[k] = dic2[k]
    return dic1


@task(returns=1)
def merge_lists(list1, list2):
    """Auxiliar method."""
    list1 = list1+list2
    return list1


# @local
def create_vocabulary(word_dic):
    """Create a partial mode."""
    word_dic = compss_wait_on(word_dic)
    docs_list = [[i[0], i[1][0], i[1][1]] for i in word_dic.items()]
    names = ['Word', 'TotalFrequency', 'DistinctFrequency']
    voc = pd.DataFrame(docs_list, columns=names)\
        .sort_values(by=['Word'])
    return voc


def filter_words(vocabulary, params):
    """Filter words."""
    min_df = params['min_df']
    min_tf = params['min_tf']
    size = params['vocab_size']
    if min_df > 0:
        vocabulary = vocabulary.loc[vocabulary['DistinctFrequency'] >= min_df]
    if min_tf > 0:
        vocabulary = vocabulary.loc[vocabulary['TotalFrequency'] >= min_tf]
    if size > 0:
        vocabulary = vocabulary.sort_values(['TotalFrequency'],
                                            ascending=False). head(size)

    return vocabulary


@task(returns=2)
def _transform_BoW(data, vocabulary, params):
    alias = params['output_col']
    columns = params['input_col']
    binary = params['binary']
    vector = np.zeros((len(data), len(vocabulary)), dtype=np.int)

    vocabulary = vocabulary['Word'].values
    data.reset_index(drop=True, inplace=True)
    for i, point in data.iterrows():
        lines = point[columns].values
        lines = np.array(list(itertools.chain(lines))).flatten()
        for e, w in enumerate(vocabulary):
            if binary:
                if w in lines:
                    vector[i][e] = 1
                else:
                    vector[i][e] = 0
            else:
                vector[i][e] = (lines == w).sum()

    data[alias] = vector.tolist()

    info = generate_info(data, 0)
    return data, info


class TfidfVectorizer(ModelDDF):
    """
    Term frequency-inverse document frequency (TF-IDF) is a numerical
    statistic transformation that is intended to reflect how important a word
    is to a document in a collection or corpus.

    :Example:

    >>> tfidf = TfidfVectorizer(input_col='col_0', output_col='col_1').fit(ddf1)
    >>> ddf2 = tfidf.transform(ddf1)
    """

    def __init__(self, input_col, output_col=None, vocab_size=-1, min_tf=1.0,
                 min_df=1.0):
        """
        :param input_col: Input column name with the tokens;
        :param output_col: Output column name;
        :param vocab_size: Maximum size of the vocabulary. If -1, no limits
         will be applied. (default, -1)
        :param min_tf: Specifies the minimum number of different documents a
         term must appear in to be included in the vocabulary. If this is an
         integer >= 1, this specifies the number of documents the term must
         appear in;  Default 1.0;
        :param min_df: Filter to ignore rare words in a document. For each
         document, terms with frequency/count less than the given threshold
         are ignored. If this is an integer >= 1, then this specifies a count
         (of times the term must appear in the document);
        """
        super(TfidfVectorizer, self).__init__()

        if not output_col:
            output_col = input_col

        self.settings = dict()
        self.settings['input_col'] = [input_col]
        self.settings['output_col'] = output_col
        self.settings['vocab_size'] = vocab_size
        self.settings['min_tf'] = min_tf
        self.settings['min_df'] = min_df

        self.model = []
        self.name = 'TfidfVectorizer'

    def fit(self, data):
        """
        Fit the model.

        :param data: DDF
        :return: trained model
        """

        vocab_size = self.settings['vocab_size']
        min_tf = self.settings['min_tf']
        min_df = self.settings['min_df']

        df, nfrag, tmp = self._ddf_inital_setup(data)

        result_p = [[] for _ in range(nfrag)]
        for f in range(nfrag):
            result_p[f] = wordCount(df[f], self.settings)
        word_dic = merge_reduce(merge_wordCount, result_p)
        vocabulary = create_vocabulary(word_dic)

        if any([min_tf > 0, min_df > 0, vocab_size > 0]):
            vocabulary = filter_words(vocabulary, self.settings)

        self.model = [vocabulary]

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
        vocabulary = self.model[0]

        df, nfrag, tmp = self._ddf_inital_setup(data)

        counts = [count_records(df[f]) for f in range(nfrag)]
        count = merge_reduce(mergeCount, counts)

        result = [[] for _ in range(nfrag)]
        info = [[] for _ in range(nfrag)]
        for f in range(nfrag):
            result[f], info[f] = \
                construct_tf_idf(df[f], vocabulary, self.settings, count)

        uuid_key = self._ddf_add_task(task_name='task_transform_tfidf',
                                      status='COMPLETED', lazy=False,
                                      function={0: result},
                                      parent=[tmp.last_uuid],
                                      n_output=1, n_input=1, info=info)

        self._set_n_input(uuid_key, 0)
        return DDF(task_list=tmp.task_list, last_uuid=uuid_key)


@task(returns=list)
def count_records(data):
    """Count the partial number of records in each fragment."""
    return len(data)


@task(returns=list)
def mergeCount(data1, data2):
    """Auxiliar method to merge the lengths."""
    return data1 + data2


@task(returns=2)
def construct_tf_idf(data, vocabulary, params, num_doc):
    """Construct the tf-idf feature.

    TF(t)  = (Number of times term t appears in a document)
                    / (Total number of terms in the document).
    IDF(t) = log( Total number of documents /
                    Number of documents with term t in it).
    Source: http://www.tfidf.com/
    """

    alias = params['output_col']
    columns = params['input_col']
    vector = np.zeros((data.shape[0], vocabulary.shape[0]), dtype=np.float)
    vocab = vocabulary['Word'].values
    data.reset_index(drop=True, inplace=True)

    for i, point in data.iterrows():
        lines = point[columns].values
        lines = np.array(list(itertools.chain(lines))).flatten()
        for w, token in enumerate(vocab):
            if token in lines:
                # TF = (Number of times term t appears in the document) /
                #        (Total number of terms in the document).
                nTimesTermT = np.count_nonzero(lines == token)
                total = len(lines)
                if total > 0:
                    tf = float(nTimesTermT) / total
                else:
                    tf = 0

                # IDF = log_e(Total number of documents /
                #            Number of documents with term t in it).
                nDocsWithTermT = vocabulary.\
                    loc[vocabulary['Word'] == token, 'DistinctFrequency'].\
                    item()
                idf = np.log(float(num_doc) / nDocsWithTermT)

                vector[i][w] = tf*idf

    data[alias] = vector.tolist()

    info = generate_info(data, 0)
    return data, info
