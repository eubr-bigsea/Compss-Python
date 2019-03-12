#!/usr/bin/python
# -*- coding: utf-8 -*-

from ddf_library.ddf import DDF
import pandas as pd


def ml_feature_text_operations():

    from ddf_library.functions.ml.feature import Tokenizer, RemoveStopWords, \
        CountVectorizer, TfidfVectorizer

    df = pd.DataFrame.from_dict({'col': ['and', 'conan', 'What', 'rare']})
    stopswords = DDF().parallelize(df, num_of_parts=2).select(['col'])

    data1 = DDF()\
        .load_text('/text_data.txt', num_of_parts=4, header=False, sep='\n')\
        .map(lambda row: row['col_0'], 'col_1')

    tokenized = Tokenizer(input_col='col_0').transform(data1)

    remover = RemoveStopWords(input_col='col_0',
                              output_col='col_1',
                              stops_words_list=['rock', 'destined'])

    remover = remover.stopwords_from_ddf(stopswords, 'col')
    result = remover.transform(tokenized)

    from ddf_library.functions.ml.feature import NGram

    result = NGram(input_col='col_1', n=2).transform(result)

    counter = CountVectorizer(input_col='col_1',
                              output_col='col_2', min_tf=0).fit(result)
    counter.save_model('/count_vectorizer')
    result1 = counter.transform(result)
    df1 = result1.cache().show()

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

    counter = TfidfVectorizer(input_col='col_0', output_col='col_2')\
        .fit(tokenized)
    counter.save_model('/tfidf_vectorizer')
    result = counter.transform(tokenized)
    df2 = result.cache().show()

    from ddf_library.functions.ml.feature import StringIndexer, IndexToString
    data = pd.DataFrame([(0, "a"), (1, "b"), (2, "c"),
                         (3, "a"), (4, "a"), (5, "c")],
                        columns=["id", "category"])

    data = DDF().parallelize(data, 4).select(['id', 'category'])

    model = StringIndexer(input_col='category').fit(data)

    converted = model.transform(data)

    result = IndexToString(input_col='category_indexed', model=model) \
        .transform(converted).drop(['id']).show()

    print df1
    print df2
    print "RESULT :", result


if __name__ == '__main__':
    print "_____Testing Text Operations_____"
    ml_feature_text_operations()
