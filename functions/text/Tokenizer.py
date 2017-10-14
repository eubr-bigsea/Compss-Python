#!/usr/bin/python
# -*- coding: utf-8 -*-


from pycompss.api.task import task
from pycompss.api.parameter import *

import pandas as pd
import numpy as np
import re


def filter_accents(s):
    return ''.join(
                    (c for c in unicodedata.normalize('NFD', s.decode('UTF-8'))
                                if unicodedata.category(c) != 'Mn')
                    )

def filter_punct (ent):
    regex = re.compile('[%s]' % re.escape(string.punctuation))
    ent = regex.sub('', ent)
    return ent

def TokenizerOperation(data,settings,numFrag):
    """
        Tokenization is the process of taking text (such as a sentence) and
        breaking it into individual terms (usually words). A simple Tokenizer
        class provides this functionality.

        :param data: A list of pandas dataframe
        :param settings: A dictionary with some configs:
            - attributes: Field with contains a text/sentence tokenized.
            - alias:      Name of the new column (default, add suffix '_tok');
            - type:       Type of the Tokenization (simple or regex).
                          In moment only "simple" is accepted;
            - min_token_length: Minimun lenght of the token (integer, default:2)
            - case-sensitive:   False to create tokens in lower case
                                (default, False)

        :return A new list of pandas dataframe
    """
    type_op = settings.get('type', "simple")
    if type_op not in ["simple"]:
        raise Exception("You must inform a valid option of `type`.")

    if 'attributes' not in settings:
        raise Exception("You must inform an `attributes` field.")

    settings['case-sensitive']   = settings.get('case-sensitive', False)
    settings['min_token_length'] = settings.get('min_token_length',2)
    settings['attributes']       = settings['attributes']
    settings['alias'] = \
                    settings.get('alias','{}_tok'.format(settings['attributes']))


    if type_op == "simple":
        result = [Tokenizer_part(data[i],settings) for i in range(numFrag)]
        return result

@task(returns=list)
def Tokenizer_part(data,settings):
    case_sensitive   = settings['case-sensitive']
    min_token_length = settings['min_token_length']
    columns          = settings['attributes']
    alias            = settings['alias']
    result = []
    for line in data[columns].values:
        row = []
        for column in line:
            toks = re.split('[?!:;\s]|(?<!\d)[,.]|[,.](?!\d)', column)
            col = []
            for t in toks:
                if len(t) > min_token_length:
                    if case_sensitive:
                        col.append(t)
                    else:
                        col.append(t.lower())
            row.append(col)
        result.append(row)

    data[alias] = np.ravel(result)
    return data
