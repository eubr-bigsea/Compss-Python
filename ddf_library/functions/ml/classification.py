#!/usr/bin/env python3
# -*- coding: utf-8 -*-


__author__ = "Lucas Miguel S Ponce"
__email__ = "lucasmsp@gmail.com"

from .classification_lib.naivebayes import GaussianNB
from .classification_lib.knn import KNearestNeighbors
from .classification_lib.logistic_regression import LogisticRegression
from .classification_lib.svm import SVM

__all__ = ['GaussianNB', 'KNearestNeighbors', 'LogisticRegression', 'SVM']
