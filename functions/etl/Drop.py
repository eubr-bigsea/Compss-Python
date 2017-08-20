#!/usr/bin/python
# -*- coding: utf-8 -*-


from pycompss.api.task import task
from pycompss.api.parameter import *


import pandas as pd



#-------------------------------------------------------------------------------
# Drop columns

def DropOperation(data, columns,numFrag):
    """
        Returns a new DataFrame that drops the specified column.
        Nothing is done if schema doesn't contain the given column name(s).
        The only parameters is the name of the columns to be removed.

        :param data: A dataframe with already splited in numFrags.
        :param settings: A list of columns
        :return: Returns a dataframe splited in numFrags.
    """

    data_result = [ Drop_part(data[f], columns) for f in range(numFrag)]

    return data_result

@task(returns=list)
def Drop_part(list1,columns):
    return  list1.drop(columns, axis=1)

#-------------------------------------------------------------------------------
