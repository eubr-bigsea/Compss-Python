#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Lucas Miguel S Ponce"
__email__ = "lucasmsp@gmail.com"


from .feature_lib.decomposition import PCA
from .feature_lib.extraction import CountVectorizer, TfidfVectorizer
from .feature_lib.preprocessing import Binarizer, OneHotEncoder, \
    StringIndexer, IndexToString, PolynomialExpansion
from .feature_lib.scalers import MinMaxScaler, MaxAbsScaler, StandardScaler
from .feature_lib.selection import VectorAssembler, VectorSlicer
from .feature_lib.text_operations import Tokenizer, RegexTokenizer, \
    RemoveStopWords, NGram

decomposition = ['PCA']
extraction = ['CountVectorizer', 'TfidfVectorizer']
preprocessing = ['Binarizer', 'StringIndexer', 'IndexToString',
                 'OneHotEncoder', 'PolynomialExpansion']
scalers = ['MaxAbsScaler', 'MinMaxScaler', 'StandardScaler']
# selection = ['VectorAssembler', 'VectorSlicer']
text_operations = ['Tokenizer', 'RegexTokenizer', 'RemoveStopWords',
                   'NGram']


__all__ = decomposition + extraction + preprocessing + \
          scalers + text_operations + text_operations

