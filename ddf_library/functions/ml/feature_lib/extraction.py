#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Lucas Miguel S Ponce"
__email__ = "lucasmsp@gmail.com"

from ddf_library.bases.context_base import ContextBase
from ddf_library.bases.metadata import OPTGroup
from ddf_library.ddf import DDF
from ddf_library.bases.ddf_model import ModelDDF
from ddf_library.utils import generate_info, read_stage_file

from pycompss.api.parameter import FILE_IN
from pycompss.api.task import task
from pycompss.functions.reduce import merge_reduce
from pycompss.api.api import compss_wait_on

import numpy as np
import pandas as pd

# TODO: 'remove' parameter for both methods


class CountVectorizer(ModelDDF):
    # noinspection PyUnresolvedReferences
    """
    Converts a collection of text documents to a matrix of token counts.

    :Example:

    >>> cv = CountVectorizer().fit(ddf1, input_col='col_1')
    >>> ddf2 = cv.transform(ddf1, output_col='col_2')
    """

    def __init__(self, vocab_size=200, min_tf=1.0, min_df=1, binary=True):
        """

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

        self.vocab_size = vocab_size
        self.min_tf = min_tf
        self.min_df = min_df
        self.binary = binary

    def fit(self, data, input_col):
        """
        Fit the model.

        :param data: DDF
        :param input_col: Input column name with the tokens;
        :return: a trained model
        """

        if isinstance(input_col, list):
            raise Exception('"input_col" must be a single column.')
        self.input_col = input_col

        df, nfrag, tmp = self._ddf_initial_setup(data)

        result_p = [[] for _ in range(nfrag)]
        for f in range(nfrag):
            result_p[f] = _wordcount(df[f], self.input_col)
        word_dic = merge_reduce(merge_word_count, result_p)

        vocabulary = create_vocabulary(word_dic, -1)

        if any([self.min_tf > 0, self.min_df > 0, self.vocab_size > 0]):
            vocabulary = filter_words(vocabulary, self.min_df, self.min_tf,
                                      self.vocab_size)

        self.model['algorithm'] = self.name
        self.model['vocabulary'] = vocabulary

        return self

    def fit_transform(self, data, input_col, output_col=None, remove=False):
        """
        Fit the model and transform.

        :param data: DDF
        :param input_col: Input column name with the tokens;
        :param output_col: Output field (default, add suffix '_vectorized');
        :param remove: Remove input columns after execution (default, False).
        :return: DDF
        """

        self.fit(data, input_col)
        ddf = self.transform(data, output_col=output_col, remove=remove)

        return ddf

    def transform(self, data, input_col=None, output_col=None, remove=False):
        """
        :param data: DDF
        :param input_col: Input column name with the tokens;
        :param output_col: Output field (default, add suffix '_vectorized');
        :param remove: Remove input columns after execution (default, False).
        :return: DDF
        """

        self.check_fitted_model()
        if input_col:
            self.input_col = input_col

        if not output_col:
            output_col = '{}_vectorized'.format(self.input_col)
        self.output_col = output_col
        self.remove = remove

        self.settings = self.__dict__.copy()

        uuid_key = ContextBase \
            .ddf_add_task(operation=self, parent=[data.last_uuid])

        return DDF(last_uuid=uuid_key)

    @staticmethod
    def function(df, params):
        params = params.copy()
        params['model'] = params['model']['vocabulary']
        return _transform_bow(df, params)


@task(returns=dict, data_input=FILE_IN)
def _wordcount(data_input, input_col):
    """Auxiliary method to create a model."""
    wordcount = {}

    data = read_stage_file(data_input, input_col)
    # first:   Number of all occurrences with term t
    # second:  Number of different documents with term t
    # third:   temporary - only to identify the last occurrence

    for i_doc, doc in enumerate(data[input_col].tolist()):

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
def merge_word_count(dic1, dic2):
    """Merge the word counts."""
    for k in dic2:
        if k in dic1:
            dic1[k][0] += dic2[k][0]
            dic1[k][1] += dic2[k][1]
        else:
            dic1[k] = dic2[k]
    return dic1


@task(returns=1)
def merge_lists(list1, list2):
    list1 = list1+list2
    return list1


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


def filter_words(vocabulary, min_df, min_tf, vocab_size):
    """Filter words."""
    if min_df > 0:
        vocabulary = vocabulary.loc[vocabulary['DistinctFrequency'] >= min_df]
    if min_tf > 0:
        vocabulary = vocabulary.loc[vocabulary['TotalFrequency'] >= min_tf]
    if vocab_size > 0:
        vocabulary = vocabulary.sort_values(['TotalFrequency'],
                                            ascending=False). head(vocab_size)

    return vocabulary


def _transform_bow(data, params):
    frag = params['id_frag']
    vocabulary = params['model']
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
    # noinspection PyTypeChecker
    vectorizer = CountVectorizer(analyzer='word',
                                 tokenizer=dummy_fun,
                                 preprocessor=dummy_fun,
                                 token_pattern=None,
                                 binary=params['binary'])
    vectorizer.vocabulary_ = vocabulary_
    vector = vectorizer.transform(data[column].tolist())

    cols = [col for col in output_col if col in data.columns]
    if len(cols) > 0:
        data.drop(cols, axis=1, inplace=True)

    vector = pd.DataFrame(vector.toarray(), columns=output_col)

    data = pd.concat([data, vector], sort=False, axis=1)

    info = generate_info(data, frag)
    return data, info


class TfidfVectorizer(ModelDDF):
    # noinspection PyUnresolvedReferences
    """
    Term frequency-inverse document frequency (TF-IDF) is a numerical
    statistic transformation that is intended to reflect how important a word
    is to a document in a collection or corpus.

    :Example:

    >>> tfidf = TfidfVectorizer(input_col='col_0', output_col='col_1').fit(ddf1)
    >>> ddf2 = tfidf.transform(ddf1)
    """

    def __init__(self, vocab_size=200, min_tf=1.0, min_df=1):
        """
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
        self.vocab_size = vocab_size
        self.min_tf = min_tf
        self.min_df = min_df

    def fit(self, data, input_col):
        """
        Fit the model.

        :param data: DDF
        :param input_col: Input column name with the tokens;
        :return: trained model
        """

        if isinstance(input_col, list):
            raise Exception('"input_col" must be a single column.')

        self.input_col = input_col

        df, nfrag, tmp = self._ddf_initial_setup(data)

        # TODO: schema instead to generate new tasks
        counts = [count_records(df[f]) for f in range(nfrag)]
        count = merge_reduce(merge_count, counts)

        result_p = [[] for _ in range(nfrag)]
        for f in range(nfrag):
            result_p[f] = _wordcount(df[f], self.input_col)
        word_dic = merge_reduce(merge_word_count, result_p)
        vocabulary = create_vocabulary(word_dic, n_rows=count)

        if any([self.min_tf > 0, self.min_df > 0, self.vocab_size > 0]):
            vocabulary = filter_words(vocabulary, self.min_df,
                                      self.min_tf, self.vocab_size)

        self.model['vocabulary'] = vocabulary
        self.model['algorithm'] = self.name

        return self

    def fit_transform(self, data, input_col, output_col=None, remove=False):
        """
        Fit the model and transform.

        :param data: DDF
        :param input_col: Input column name with the tokens;
        :param output_col: Output field (default, add suffix '_vectorized');
        :param remove: Remove input columns after execution (default, False).
        :return: DDF
        """

        self.fit(data, input_col)
        ddf = self.transform(data, output_col)

        return ddf

    def transform(self, data, input_col=None, output_col=None, remove=False):
        """
        :param data: DDF
        :param input_col: Input column name with the tokens;
        :param output_col: Output field (default, add suffix '_vectorized');
        :param remove: Remove input columns after execution (default, False).
        :return: DDF
        """

        self.check_fitted_model()
        if input_col:
            self.input_col = input_col

        if not output_col:
            output_col = '{}_vectorized'.format(self.input_col)
        self.output_col = output_col
        self.remove = remove

        self.settings = self.__dict__.copy()

        uuid_key = ContextBase \
            .ddf_add_task(operation=self, parent=[data.last_uuid])

        return DDF(last_uuid=uuid_key)

    @staticmethod
    def function(df, params):
        params = params.copy()
        params['model'] = params['model']['vocabulary']
        return construct_tf_idf(df, params)


@task(returns=1, data_input=FILE_IN)
def count_records(data_input):
    """Count the partial number of records in each fragment."""
    data = read_stage_file(data_input)
    return len(data)


@task(returns=1)
def merge_count(data1, data2):
    """Auxiliary method to merge the lengths."""
    return data1 + data2


def construct_tf_idf(data, params):
    """Construct the tf-idf feature.

    TF(t)  = (Number of times term t appears in a document)
                    / (Total number of terms in the document).
    IDF(t) = log( Total number of documents /
                    Number of documents with term t in it).
    Source: http://www.tfidf.com/
    """

    frag = params['id_frag']
    vocabulary = params['model']
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
    # noinspection PyTypeChecker
    vectorizer = TfidfVectorizer(analyzer='word',
                                 tokenizer=dummy_fun,
                                 preprocessor=dummy_fun,
                                 token_pattern=None)
    vectorizer.vocabulary_ = vocabulary_
    vectorizer.idf_ = vocabulary['InverseDistinctFrequency'].values
    vector = vectorizer.transform(data[column].tolist())

    cols = [col for col in output_col if col in data.columns]
    if len(cols) > 0:
        data.drop(cols, axis=1, inplace=True)

    vector = pd.DataFrame(vector.toarray(), columns=output_col)
    data = pd.concat([data, vector], sort=False, axis=1)

    info = generate_info(data, frag)
    return data, info
