#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__ = "Lucas Miguel S Ponce"
__email__  = "lucasmsp@gmail.com"

from pycompss.api.task import task
from pycompss.api.parameter import *


def DropOperation(data, columns, numFrag):
    """
        DropOperation():
        Returns a new DataFrame that drops the specified column.
        Nothing is done if schema doesn't contain the given column name(s).

        :param data:    A list with numFrag pandas's dataframe;
        :param columns: A list with the columns names to be removed;
        :param numFrag: A number of fragments;
        :return: A list with numFrag pandas's dataframe.
    """
    result = [[] for f in range(numFrag)]
    for f in range(numFrag):
        result[f] = Drop_part(data[f], columns)

    return result

@task(returns=list)
def Drop_part(list1,columns):
    import pandas as pd
    return  list1.drop(columns, axis=1)
