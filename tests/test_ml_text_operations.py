#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from ddf_library.ddf import DDF
import pandas as pd


def tokenizer():
    from ddf_library.functions.ml.feature import Tokenizer

    data1 = DDF() \
        .load_text('/text_data.txt', num_of_parts=4, header=False, sep='\n')

    tokenized = Tokenizer(input_col='col_0').transform(data1)

    return tokenized


def remove_stopwords(ddf1):

    from ddf_library.functions.ml.feature import RemoveStopWords

    df = pd.DataFrame.from_dict({'col': ['and', 'conan', 'What', 'rare']})
    stopswords = DDF().parallelize(df, num_of_parts=2)

    remover = RemoveStopWords(input_col='col_0_tokenized',
                              output_col='col_1',
                              stops_words_list=['rock', 'destined'])

    result = remover.stopwords_from_ddf(stopswords, 'col').transform(ddf1)

    return result


def ngram(ddf1):

    from ddf_library.functions.ml.feature import NGram

    result = NGram(input_col='col_1', output_col='col_1', n=2).transform(ddf1)

    return result


def vectorize(ddf1):
    from ddf_library.functions.ml.feature import CountVectorizer, \
        TfidfVectorizer

    counter = CountVectorizer(input_col='col_1',
                              output_col='col_2', min_tf=0).fit(result)
    counter.save_model('/count_vectorizer')
    result1 = counter.transform(result)
    result1.show()


def TfidfVectorizer():

    from ddf_library.functions.ml.feature import TfidfVectorizer, Tokenizer
    corpus = [
             'This is the first document',
             'This document is the second document',
             'And this is the third one',
             'Is this the first document',
        ]
    df = pd.DataFrame.from_dict({'col_0': corpus})

    test_data = DDF().parallelize(df, num_of_parts=2)

    tokenized = Tokenizer(input_col='col_0',
                          min_token_length=1).transform(test_data)

    counter = TfidfVectorizer(input_col='col_0_tokenized', output_col='col_2')\
        .fit(tokenized)
    counter.save_model('/tfidf_vectorizer')
    result = counter.transform(tokenized)
    result.show()
    pass


def categorization():

    from ddf_library.functions.ml.feature import StringIndexer, IndexToString
    data = pd.DataFrame([(0, "a"), (1, "b"), (2, "c"),
                         (3, "a"), (4, "a"), (5, "c")],
                        columns=["id", "category"])

    data = DDF().parallelize(data, 4).select(['id', 'category'])

    model = StringIndexer(input_col='category').fit(data)

    converted = model.transform(data)

    result = IndexToString(input_col='category_indexed', model=model) \
        .transform(converted).drop(['id']).show()

    result.show()


if __name__ == '__main__':
    print("_____Testing Text Operations_____")
    # ddf1 = tokenizer()
    # ddf2 = remove_stopwords(ddf1)
    # ddf3 = ngram(ddf2)
    TfidfVectorizer()
    # categorization()
