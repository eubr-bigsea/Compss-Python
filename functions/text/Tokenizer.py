#!/usr/bin/python
# -*- coding: utf-8 -*-


from pycompss.api.task import task
from pycompss.api.parameter import *

import pandas as pd
import numpy as np
import re


def filter_accents(s):
    return ''.join(
        (c for c in unicodedata.normalize('NFD', s.decode('UTF-8')) if unicodedata.category(c) != 'Mn'))

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
                         - type: Type of the Tokenization (simple or regex). But
                         in moment only "simple" is accepted.
                         - min_token_length: minimun lenght of the token

        :return A new list  of pandas dataframe
    """
    type_op = settings.get('type',"simple")
    if type_op == "simple":
        result = [Tokenizer_part(data[i],settings) for i in range(numFrag)]
        return result

@task(returns=list)
def Tokenizer_part(data,settings):
    case_sensitive   = settings.get('case-sensitive', False)
    min_token_length = int(settings['min_token_length'])
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

    #data[alias] = pd.Series(result).values
    #return data
    tmp = pd.DataFrame(result, columns=alias)
    result = pd.concat([data.reset_index(drop=True), tmp], axis=1)

    return result
