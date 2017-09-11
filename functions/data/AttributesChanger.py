# -*- coding: utf-8 -*-
#!/usr/bin/env python


from pycompss.api.task          import task
from pycompss.api.parameter     import *


import numpy as np
import pandas as pd



def AttributesChangerOperation(data,settings,numFrag):
    """
    AttributesChangerOperation():

    Rename or change the data's type of some columns.

    :param data:       A list with numFrag pandas's dataframe;
    :param settings:   A dictionary that contains:
        - attributes:  The column(s) to be changed (or renamed).
        - new_name:    The new name of the column. If used with multiple
                       attributes, a numerical suffix will be used to
                       distinguish them.
        - new_data_type: The new type of the selected columns:
            * 'keep';
            * 'integer';
            * 'string';
            * 'double';
            * 'Date';
            * 'Date/time';
    :param numFrag:    The number of fragments;
    :return:           Returns a list with numFrag pandas's dataframe.
    """

    attributes = settings.get('attributes',[])
    if len(attributes)==0:
        return data
    new_data_type = settings.get('new_data_type','keep')
    new_name      = settings.get('new_name',[])

    result = [changeAttribute(data[f],attributes,new_name,new_data_type) for f in range(numFrag)]

    return result

@task(returns=list)
def changeAttribute(data,attributes,new_name,new_data_type):

    if len(new_name) == 0:
        new_name = attributes
    elif len(attributes) > 1:
        new_name = [ "{}_{}".format(new_name[0],d) for d in range(len(attributes))]


    mapper =  dict(zip(attributes, new_name))

    if new_data_type == 'keep':
        pass
    elif new_data_type == 'integer':
        data[attributes] = pd.to_numeric(data[attributes], downcast='integer', errors='coerce') #.astype(int)
    elif new_data_type == 'string':
        data[attributes] = data[attributes].astype(str)
    elif new_data_type == "double":
        data[attributes] = pd.to_numeric(data[attributes], downcast='float', errors='coerce')
    elif new_data_type == "Date":
        from dateutil import parser
        f = lambda col:  parser.parse(col).date()
        for a in attributes:
            data[a] =  data[a].apply(f)
    elif new_data_type == "Date/time":
        from dateutil import parser
        f = lambda col: parser.parse(col)
        for a in attributes:
            data[a] =  data[a].apply(f)

    data.rename(columns=mapper, inplace=True)
    return data
