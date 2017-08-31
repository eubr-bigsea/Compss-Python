# Bag-of-words (BoW):
The bag-of-words model is a simplifying representation used in natural language processing and information retrieval. In this model, a text (such as a sentence or a document) is represented as the bag (multiset) of its words, disregarding grammar and even word order but keeping multiplicity.

## Instructions:

First, use the method `fit()` to create a model (which contains a vocabulary) based on the training dataset. After that, use the method `transform()` to append a new column (or replace it) with the transformed documents.

All parameters are explained below:

**fit()**: Create a dictionary (vocabulary) of each word and its frequency in this set and in how many documents occured.
* :param train_set: A list of pandas dataframe with the documents to be transformed.
* :param params:    A dictionary with some options:
    - minimum_df:   Minimum number of how many  documents a word should appear.
    - minimum_tf:    Minimum number of occurrences of a word.
    - size:         Vocabulary maximum size, -1 if there are no size.
* :param numFrag: num fragments, if -1 data is considered chunked.
* :return  A model (dataframe) with the <word,tf,df>

**transform()**: Perform the transformation of the data based in the model created.
* :param test_set:  A list of dataframes with the documents to transform;
* :param vocabulary:  A model trained (grammar and its frequency);
* :param params: A dictionary with the settings:
                            - alias: new name of the column;
                            - attributes: all columns which contains the text. Each row is considered a document.
* :param numFrag:   The number of fragments;
* :return   A list of pandas dataframe with the features transformed.

## Example:


```sh
from BagOfWords import *
settings = dict()
settings['attributes']  = 'column'
settings['alias']       = 'Result'
settings['minimum_df'] = 10
settings['minimum_tf'] = 100
settings['size'] = -1
BoW = BagOfWords()
model  = BoW.fit(data,settings,numFrag)
result = BoW.transform(data, model, settings, numFrag)
``` 
