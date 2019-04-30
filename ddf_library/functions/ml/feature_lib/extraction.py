#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Lucas Miguel S Ponce"
__email__ = "lucasmsp@gmail.com"

from ddf_library.ddf import DDF, DDFSketch
from ddf_library.ddf_model import ModelDDF
from ddf_library.utils import generate_info


from pycompss.api.task import task
from pycompss.functions.reduce import merge_reduce
from pycompss.api.api import compss_wait_on
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

    def __init__(self, input_col, vocab_size=200, min_tf=1.0,
                 min_df=1, binary=True):
        """
        :param input_col: Input column name with the tokens;
        :param vocab_size: Maximum size of the vocabulary. (default, 200)
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

        if isinstance(input_col, list):
            raise Exception('"input_col" must be a single column.')

        self.settings = dict()
        self.settings['input_col'] = input_col
        self.settings['vocab_size'] = vocab_size
        self.settings['min_tf'] = min_tf
        self.settings['min_df'] = min_df
        self.settings['binary'] = binary

        self.name = 'CountVectorizer'
        self.model = dict()

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
            result_p[f] = _wordcount(df[f], self.settings)
        word_dic = merge_reduce(merge_wordCount, result_p)

        vocabulary = create_vocabulary(word_dic, -1)

        if any([min_tf > 0, min_df > 0, vocab_size > 0]):
            vocabulary = filter_words(vocabulary, self.settings)

        self.model['algorithm'] = self.name
        self.model['vocabulary'] = vocabulary

        return self

    def fit_transform(self, data, output_col=None):
        """
        Fit the model and transform.

        :param data: DDF
        :param output_col: Output field (default, add suffix '_vectorized');
        :return: DDF
        """

        self.fit(data)
        ddf = self.transform(data, output_col)

        return ddf

    def transform(self, data, output_col=None):
        """
        :param data: DDF
        :param output_col: Output field (default, add suffix '_vectorized');
        :return: DDF
        """
        if len(self.model) == 0:
            raise Exception("Model is not fitted.")

        if output_col is not None:
            self.settings['output_col'] = output_col

        vocabulary = self.model['vocabulary']

        df, nfrag, tmp = self._ddf_inital_setup(data)

        result = [[] for _ in range(nfrag)]
        info = [[] for _ in range(nfrag)]
        for f in range(nfrag):
            result[f], info[f] = _transform_bow(df[f], vocabulary,
                                                self.settings, f)

        uuid_key = self._ddf_add_task(task_name='transform_count_vectorizer',
                                      status='COMPLETED', lazy=False,
                                      function={0: result},
                                      parent=[tmp.last_uuid],
                                      n_output=1, n_input=1, info=info)

        self._set_n_input(uuid_key, 0)
        return DDF(task_list=tmp.task_list, last_uuid=uuid_key)


@task(returns=dict)
def _wordcount(data, params):
    """Auxilar method to create a model."""
    wordcount = {}
    columns = params['input_col']
    # first:   Number of all occorrences with term t
    # second:  Number of diferent documents with term t
    # third:   temporary - only to idetify the last occorrence

    for i_doc, doc in enumerate(data[columns].tolist()):

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
def create_vocabulary(word_dic, n_rows=-1):
    """Create a partial mode."""
    word_dic = compss_wait_on(word_dic)

    names = ['Word', 'TotalFrequency', 'DistinctFrequency']
    docs_list = [[i[0], i[1][0], i[1][1]] for i in word_dic.items()]
    voc = pd.DataFrame(docs_list, columns=names).sort_values(by=['Word'])

    if n_rows != -1:
        n_rows = compss_wait_on(n_rows)

        # smooth_idf: Smooth idf weights by adding one to document
        # frequencies, as if an extra document was seen containing every term in
        # the collection exactly once.Prevents zero divisions.

        # IDF = log_e(Total number of documents /
        #            Number of documents containing t).
        voc['InverseDistinctFrequency'] = \
            np.log((n_rows + 1)/(voc['DistinctFrequency']+1)) + 1

    voc.reset_index(drop=True, inplace=True)
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
def _transform_bow(data, vocabulary, params, frag):

    vocabulary_ = {k: v for v, k in enumerate(vocabulary['Word'].values)}
    n_cols = len(vocabulary_)
    column = params['input_col']
    output_col = params.get('output_col',
                            ['vec_{}'.format(i) for i in range(n_cols)])

    if not isinstance(output_col, list):
        output_col = ['{}{}'.format(output_col, i) for i in range(n_cols)]

    data.reset_index(drop=True, inplace=True)

    def dummy_fun(doc):
        return doc

    from sklearn.feature_extraction.text import CountVectorizer
    vectorizer = CountVectorizer(analyzer='word',
                                 tokenizer=dummy_fun,
                                 preprocessor=dummy_fun,
                                 token_pattern=None,
                                 binary=params['binary'])
    vectorizer.vocabulary_ = vocabulary_
    vector = vectorizer.transform(data[column].tolist())

    # vector = np.zeros((len(data), len(vocabulary)), dtype=np.int)
    # if len(data) > 0:
    #     for i, lines in enumerate(data[column].tolist()):
    #         for e, w in enumerate(vocabulary):
    #             if binary:
    #                 if w in lines:
    #                     vector[i][e] = 1
    #                 else:
    #                     vector[i][e] = 0
    #             else:
    #                 vector[i][e] = (lines == w).sum()

    cols = [col for col in output_col if col in data.columns]
    if len(cols) > 0:
        data.drop(cols, axis=1, inplace=True)

    vector = pd.DataFrame(vector.toarray(), columns=output_col)

    data = pd.concat([data, vector], sort=False, axis=1)

    info = generate_info(data, frag)
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

    def __init__(self, input_col, vocab_size=200, min_tf=1.0, min_df=1):
        """
        :param input_col: Input column name with the tokens;
        :param vocab_size: Maximum size of the vocabulary. (default, 200)
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

        if isinstance(input_col, list):
            raise Exception('"input_col" must be a single column.')

        self.settings = dict()
        self.settings['input_col'] = input_col
        self.settings['vocab_size'] = vocab_size
        self.settings['min_tf'] = min_tf
        self.settings['min_df'] = min_df

        self.model = dict()
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

        # TODO: info instead to generate new tasks
        counts = [count_records(df[f]) for f in range(nfrag)]
        count = merge_reduce(mergeCount, counts)

        result_p = [[] for _ in range(nfrag)]
        for f in range(nfrag):
            result_p[f] = _wordcount(df[f], self.settings)
        word_dic = merge_reduce(merge_wordCount, result_p)
        vocabulary = create_vocabulary(word_dic, n_rows=count)

        if any([min_tf > 0, min_df > 0, vocab_size > 0]):
            vocabulary = filter_words(vocabulary, self.settings)

        self.model['vocabulary'] = vocabulary
        self.model['algorithm'] = self.name

        return self

    def fit_transform(self, data, output_col=None):
        """
        Fit the model and transform.

        :param data: DDF
        :param output_col: Output field (default, add suffix '_vectorized');
        :return: DDF
        """

        self.fit(data)
        ddf = self.transform(data, output_col)

        return ddf

    def transform(self, data, output_col=None):
        """
        :param data: DDF
        :param output_col: Output field (default, add suffix '_vectorized');
        :return: DDF
        """

        if len(self.model) == 0:
            raise Exception("Model is not fitted.")

        if output_col is not None:
            self.settings['output_col'] = output_col

        vocabulary = self.model['vocabulary']

        df, nfrag, tmp = self._ddf_inital_setup(data)

        result = [[] for _ in range(nfrag)]
        info = [[] for _ in range(nfrag)]
        for f in range(nfrag):
            result[f], info[f] = \
                construct_tf_idf(df[f], vocabulary, self.settings, f)

        uuid_key = self._ddf_add_task(task_name='task_transform_tfidf',
                                      status='COMPLETED', lazy=False,
                                      function={0: result},
                                      parent=[tmp.last_uuid],
                                      n_output=1, n_input=1, info=info)

        self._set_n_input(uuid_key, 0)
        return DDF(task_list=tmp.task_list, last_uuid=uuid_key)


@task(returns=1)
def count_records(data):
    """Count the partial number of records in each fragment."""
    return len(data)


@task(returns=1)
def mergeCount(data1, data2):
    """Auxiliar method to merge the lengths."""
    return data1 + data2


@task(returns=2)
def construct_tf_idf(data, vocabulary, params, frag):
    """Construct the tf-idf feature.

    TF(t)  = (Number of times term t appears in a document)
                    / (Total number of terms in the document).
    IDF(t) = log( Total number of documents /
                    Number of documents with term t in it).
    Source: http://www.tfidf.com/
    """

    vocabulary_ = {k: v for v, k in enumerate(vocabulary['Word'].values)}
    n_cols = len(vocabulary_)
    column = params['input_col']
    output_col = params.get('output_col',
                            ['vec_{}'.format(i) for i in range(n_cols)])

    if not isinstance(output_col, list):
        output_col = ['{}{}'.format(output_col, i) for i in range(n_cols)]

    data.reset_index(drop=True, inplace=True)

    def dummy_fun(doc):
        return doc

    from sklearn.feature_extraction.text import TfidfVectorizer
    vectorizer = TfidfVectorizer(analyzer='word',
                                 tokenizer=dummy_fun,
                                 preprocessor=dummy_fun,
                                 token_pattern=None)
    vectorizer.vocabulary_ = vocabulary_
    vectorizer.idf_ = vocabulary['InverseDistinctFrequency'].values
    vector = vectorizer.transform(data[column].tolist())

    # vector = np.zeros((data.shape[0], vocabulary.shape[0]), dtype=np.float)
    # if len(data) > 0:
    #     vocab = vocabulary['Word'].values
    #     for i, lines in enumerate(data[column].tolist()):
    #         total = len(lines)
    #         for w, token in enumerate(vocab):
    #             if token in lines:
    #                 # TF = (Number of times term t appears in the document) /
    #                 #        (Total number of terms in the document).
    #                 tf = np.count_nonzero(lines == token) / total
    #
    #                 idf = vocabulary.\
    #                     loc[vocabulary['Word'] == token,
    #                         'InverseDistinctFrequency'].item()
    #
    #                 vector[i][w] = tf*idf

    cols = [col for col in output_col if col in data.columns]
    if len(cols) > 0:
        data.drop(cols, axis=1, inplace=True)

    vector = pd.DataFrame(vector.toarray(), columns=output_col)
    data = pd.concat([data, vector], sort=False, axis=1)

    info = generate_info(data, frag)
    return data, info
