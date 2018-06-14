#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__ = "Lucas Miguel S Ponce"
__email__ = "lucasmsp@gmail.com"

from pycompss.api.task import task
import numpy as np
import re


class TokenizerOperation(object):
    """Tokenizer Operation.

    Tokenization is the process of taking text (such as a sentence) and
    breaking it into individual terms (usually words). A simple Tokenizer
    class provides this functionality.
    """

    def transform(self, data, settings, nfrag):
        """TokenizerOperation.

        :param data: A list of pandas dataframe
        :param settings: A dictionary with some configs:
            - attributes: List with field(s) with contains
                a text/sentence to be tokenized.
            - alias: Name(s) of the new column (default, add suffix '_tok');
            - type: Type of the Tokenization (simple or regex);
            - expression: Regex expression (only if type is 'regex');
            - min_token_length: Minimun lenght of the token (integer, default:2)
            - case-sensitive: False to create tokens in lower case
                (default, False)
        :param nfrag: A number of fragments
        :return A new list of pandas dataframe

        Note: If alias is not a list, all the tokenized
                fields will be merged in one unique column.
        """
        settings = self.preprocessing(settings)
        result = [[] for _ in range(nfrag)]
        for f in range(nfrag):
            result[f] = _tokenizer(data[f], settings)
        return result

    def preprocessing(self, settings):
        """Check and add missing fields."""
        type_op = settings.get('type', "simple")
        if type_op not in ["simple", "regex"]:
            raise Exception("You must inform a valid option of `type`.")

        if 'attributes' not in settings:
            raise Exception("You must inform an `attributes` field.")

        if type_op == 'regex':
            if 'expression' not in settings:
                raise Exception(
                        "You must inform an `expression` to use regex.")

        settings['case-sensitive'] = settings.get('case-sensitive', False)
        settings['min_token_length'] = settings.get('min_token_length', 2)
        attributes = settings['attributes']

        if isinstance(attributes, list):
            alias = ['{}_tok'.format(col) for col in attributes]
            alias = settings.get('alias', alias)
            if isinstance(alias, list):
                # Adjust alias in order to have the same number of aliases
                # as attributes by filling missing alias with the attribute
                # name sufixed by _tok.
                from itertools import izip_longest
                alias = [x[1] or '{}_tok'.format(x[0]) for x in
                         izip_longest(attributes, alias[:len(attributes)])]
        else:
            raise Exception("The Parameter `attributes` must be a list.")

        settings['alias'] = alias

        return settings

    def transform_serial(self, data, settings):
        """Perform a partial tokenizer."""
        return _tokenizer_(data, settings)


@task(returns=list)
def _tokenizer(data, settings):
    return _tokenizer_(data, settings)


def _tokenizer_(data, settings):
    """Perform a partial tokenizer."""
    case_sensitive = settings['case-sensitive']
    min_token_length = settings['min_token_length']
    columns = settings['attributes']
    alias = settings['alias']
    expression = settings.get('expression',
                              '[?!:;\s]|(?<!\d)[,.]|[,.](?!\d)')
    result = []
    for field in data[columns].values:
        row = []
        for sentence in field:
            toks = re.split(expression, sentence)
            col = []
            for t in toks:
                if len(t) > min_token_length:
                    if case_sensitive:
                        col.append(t)
                    else:
                        col.append(t.lower())
            row.append(col)
        result.append(row)

    if isinstance(alias, list):
        for i, col in enumerate(alias):
            tmp = np.array(result)[:, i]
            if len(tmp) > 0:
                data[col] = tmp
            else:
                data[col] = np.nan
    else:
        data[alias] = np.ravel(result)

    return data
