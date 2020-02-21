#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from ddf_library.context import COMPSsContext
import pandas as pd


def tokenizer(cc):
    from ddf_library.functions.ml.feature import Tokenizer

    corpus = [
        'This is the first document',
        'This document is the second document',
        'And this is the third one',
        'Is this the first document',
    ]
    data1 = pd.DataFrame.from_dict({'col_0': corpus})

    data1 = cc.parallelize(data1)

    tokens = Tokenizer().transform(data1, input_col='col_0')
    tokens.show()
    return tokens


def remove_stopwords(cc):
    from ddf_library.functions.ml.feature import RemoveStopWords

    ddf1 = tokenizer(cc)
    remover = RemoveStopWords(stops_words_list=['rock', 'destined', 'third'],
                              language='english')

    result = remover\
        .transform(ddf1, input_col='col_0_tokens', output_col='col_1')

    result.show()


def ngram(cc):
    from ddf_library.functions.ml.feature import NGram

    ddf1 = tokenizer(cc)
    result = NGram(n=2).transform(ddf1, input_col='col_0_tokens',
                                  output_col='col_0_tokens')

    result.show()


def count_vectorizer(cc):
    from ddf_library.functions.ml.feature import CountVectorizer, Tokenizer
    corpus = [
             'This is the first document',
             'This document is the second document',
             'And this is the third one',
             'Is this the first document',
        ]
    df = pd.DataFrame.from_dict({'col_0': corpus})

    test_data = cc.parallelize(df, num_of_parts=2)

    dff_tokens = Tokenizer(min_token_length=1).transform(test_data,
                                                         input_col='col_0')

    dff_tokens.show()

    counter = CountVectorizer().fit(dff_tokens, input_col='col_0_tokens')

    # counter.save_model('/tfidf_vectorizer')
    result = counter.transform(dff_tokens, output_col='vec_')
    result.show()


def tf_idf_vectorizer(cc):

    from ddf_library.functions.ml.feature import TfidfVectorizer, Tokenizer
    corpus = [
             'This is the first document',
             'This document is the second document',
             'And this is the third one',
             'Is this the first document',
        ]
    df = pd.DataFrame.from_dict({'col_0': corpus})

    test_data = cc.parallelize(df, num_of_parts=2)

    dff_tokens = Tokenizer(min_token_length=1).transform(test_data,
                                                         input_col='col_0')

    counter = TfidfVectorizer().fit(dff_tokens, input_col='col_0_tokens')

    print(counter.model['vocabulary'])
    # counter.save_model('/tfidf_vectorizer')
    result = counter.transform(dff_tokens,  output_col='col_2')
    result.show()


def categorization(cc):

    from ddf_library.functions.ml.feature import StringIndexer, IndexToString
    data = pd.DataFrame([(0, "a", 'b'), (1, "b", 'b'), (2, "c", 'b'),
                         (3, "a", 'b'), (4, "a", 'b'), (5, "c", 'b')],
                        columns=["id", "category", 'category2'])

    data = cc.parallelize(data, 4)

    model = StringIndexer().fit(data, input_col=['category', 'category2'])
    converted = model.transform(data)

    result = IndexToString(model=model) \
        .transform(converted, input_col=['category_indexed',
                                         'category2_indexed']).drop(['id'])

    result.show()


if __name__ == '__main__':
    print("_____Testing Text Operations_____")
    import argparse

    parser = argparse.ArgumentParser(
            description="Testing Text Operations")
    parser.add_argument('-o', '--operation',
                        type=int,
                        required=True,
                        help="""
                        1. Tokenizer
                        2. TF-IDF vectorizer
                        3. Count Vectorizer
                        4. Categorization
                        5. Remove stopwords
                        6. nGram
                        """)
    arg = vars(parser.parse_args())

    operation = arg['operation']
    list_operations = [tokenizer,
                       tf_idf_vectorizer,
                       count_vectorizer,
                       categorization,
                       remove_stopwords,
                       ngram]

    cc = COMPSsContext()
    list_operations[operation - 1](cc)
    cc.stop()
