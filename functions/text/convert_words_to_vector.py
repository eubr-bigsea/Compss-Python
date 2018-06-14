#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__ = "Lucas Miguel S Ponce"
__email__ = "lucasmsp@gmail.com"


class ConvertWordstoVectorOperation(object):
    """Convert Words to Vector.

    - Bag of words (BoW)
    - Term frequency-inverse document frequency (TF-IDF)
    """
    def preprocessing(self, data, params, nfrag):
        mode = params.get('mode', 'BoW')
        if mode not in ['BoW', 'TF-IDF']:
            raise Exception("You must inform a valid mode to convert "
                            "your text into vectors")
        if mode == 'BoW':
            from bag_of_words import BagOfWords
            model = BagOfWords().fit(data, params, nfrag)
            return params, model
        elif mode == 'TF-IDF':
            from tf_idf import TfIdf
            model = TfIdf().fit(data, params, nfrag)
            count, params = TfIdf().preprocessing(data, params)
            params['num_docs'] = count
            return params, model

    def transform(self, data, vocabulary, params, nfrag):
        """ConvertWordstoVectorOperation.

        :param data: A list of pandas dataframe with the documents
            to be transformed.
        :param vocabulary: The created model in preprocessing stage
        :param params: A dictionary with some options:
            - mode: 'BoW' to use Bag-Of-Words (default),
                'TF-IDF' to use Term frequency  inverse document frequency;
            - all other specific parameters of each of the algorithms;
        :param nfrag: A number of fragments
        :return The new dataframe with the transformed data and a model.
        """
        mode = params.get('mode', 'BoW')
        if mode == 'BoW':
            from bag_of_words import BagOfWords
            data = BagOfWords().transform(data, vocabulary, params, nfrag)
            return data
        elif mode == 'TF-IDF':
            from tf_idf import TfIdf
            data = TfIdf().transform(data, vocabulary, params, nfrag)
            return data

    def transform_serial(self, data, vocabulary, params):
        """ConvertWordstoVectorOperation.

        :param data: A list of pandas dataframe with the documents
            to be transformed.
        :param vocabulary: The created model in preprocessing stage
        :param params: A dictionary with some options:
            - mode: 'BoW' to use Bag-Of-Words (default),
                'TF-IDF' to use Term frequency  inverse document frequency;
            - all other specific parameters of each of the algorithms;
        :param nfrag: A number of fragments
        :return The new dataframe with the transformed data and a model.
        """
        mode = params.get('mode', 'BoW')
        if mode == 'BoW':
            from bag_of_words import BagOfWords
            data = BagOfWords().transform_serial(data, vocabulary, params)
            return data
        elif mode == 'TF-IDF':
            from tf_idf import TfIdf
            data = TfIdf().transform_serial(data, vocabulary, params)
            return data



