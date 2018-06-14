#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__ = "Lucas Miguel S Ponce"
__email__ = "lucasmsp@gmail.com"

from pycompss.api.task import task
from pycompss.functions.reduce import mergeReduce
import numpy as np


class RemoveStopWordsOperation(object):
    """RemoveStopWordsOperation.

    Stop words are words which should be excluded from the input,
    typically because the words appear frequently and don’t carry
     as much meaning.
    
    """

    def transform(self, data, stopwords, settings, nfrag):
        """
            :param data: A list of pandas dataframe;
            :param stopwords: A list of pandas dataframe with stopwords,
                one token by row (empty to don't use it);
            :param settings:  A dictionary with:
                - news-stops-words: A list with some stopwords (default, [])
                - case-sensitive: True or False (default, True)
                - attributes: A list with columns which contains the tokenized
                     text/sentence;
                - alias: Name of the new column (default, tokenized_rm);
                - col_words: Attribute of second data source with stop words;
            :return  Returns a list of pandas dataframe
        """
        settings, stopwords = self.preprocessing(stopwords, settings, nfrag)

        result = [[] for _ in range(nfrag)]
        for f in range(nfrag):
            result[f] = _remove_stopwords(data[f], settings, stopwords)
        return result

    def preprocessing(self, stopwords, settings, nfrag):

        if 'attributes' not in settings:
            raise Exception("You must inform an `attributes` column.")

        settings['news-stops-words'] = settings.get('news-stops-words', [])
        settings['case-sensitive'] = settings.get('case-sensitive', True)
        settings['alias'] = settings.get('alias', "tokenized_rm")

        len_stopwords = len(stopwords)
        if len_stopwords == 0:
            stopwords = [[]]
        else:
            if 'col_words' not in settings:
                raise Exception("You must inform an `col_words` column.")

        # It assumes that stopwords's dataframe can fit in memmory
        for f in range(nfrag):
            stopwords[f] = read_stopwords(stopwords[f], settings)
        stopwords = mergeReduce(merge_stopwords, stopwords)
        return settings, stopwords

    def transform_serial(self, data, settings, stopwords):
        return _remove_stopwords_(data, settings, stopwords)


@task(returns=list)
def read_stopwords(data1, settings):
    if len(data1) > 0:
        data1 = np.reshape(data1[settings['col_words']], -1, order='C')
    else:
        data1 = np.array([])
    return data1

@task(returns=list)
def merge_stopwords(data1, data2):
    data1 = np.concatenate((data1, data2), axis=0)
    return data1

@task(returns=list)
def _remove_stopwords(data, settings, stopwords):
    return _remove_stopwords_(data, settings, stopwords)

def _remove_stopwords_(data, settings, stopwords):
    """Remove stopwords from a column."""
    columns = settings['attributes']
    alias   = settings['alias']

    # stopwords must be in 1-D
    new_stops = np.reshape(settings['news-stops-words'], -1, order='C')
    if len(stopwords) != 0:
        stopwords = np.concatenate((stopwords, new_stops), axis=0)
    else:
        stopwords = new_stops

    new_data = []
    if data.shape[0] > 0:
        if settings['case-sensitive']:
            stopwords = set(stopwords)
            for index, row in data.iterrows():
                col = []
                for entry in row[columns]:
                    col.append(list(set(entry).difference(stopwords)))
                new_data.append(col)

        else:
            stopwords = [tok.lower() for tok in stopwords]
            stopwords = set(stopwords)

            for index, row in data.iterrows():
                col = []
                for entry in row[columns]:
                    entry = [tok.lower() for tok in entry]
                    col.append(list(set(entry).difference(stopwords)))
                new_data.append(col)

        data[alias] = np.reshape(new_data, -1, order='C')
    return data
