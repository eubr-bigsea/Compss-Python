# -*- coding: utf-8 -*-
#!/usr/bin/env python


from pycompss.functions.data import chunks
from pycompss.api.task          import task
from pycompss.api.parameter     import *
from pycompss.functions.reduce import mergeReduce

import numpy as np
import math
import pickle
import pandas as pd


#------------------------------------------------------------------------------
# Load Methods



def ReadParallelFile(filename,separator,header,infer,na_values):
    import os, os.path


    data = []
    DIR = filename+"_folder"


    #get the number of files in this folder
    data =  [ ReadFromFile(DIR+"/"+name,separator,header,infer,na_values)
                for name in sorted(os.listdir(DIR)) ]

    return data

@task(returns=list, filename = FILE_IN)
def ReadFromFile(filename,separator,header,infer,na_values):
    """
        ReadFromFile:

        Method used to load a data from a file to an np.array.

        :param filename: The name of the file.
        :param separator: The string used to separate values.
        :param list_columns: A list with which columns to read
                            (with 0 being the first).
        :return A np.array with the data loaded.
    """
    if separator == "<new_line>": separator = "\n"

    if infer =="NO":
        if header:
            df = pd.read_csv(filename,sep=separator,na_values=na_values,dtype='str');
        else:
            df = pd.read_csv(filename,sep=separator,na_values=na_values,header=0,dtype='str');

    elif infer == "FROM_VALUES":
        if header:
            df = pd.read_csv(filename,sep=separator,na_values=na_values);
        else:
            df = pd.read_csv(filename,sep=separator,na_values=na_values,header=0);

    return df



def ReadFromNumpy(infile):
    """
        Load an array from a binary file in NumPy .npy format

        :param infile: the /path/file.npy
    """
    return np.load(infile)

def ReadFromPickle(infile):
    """
        Load an array from a serizable Pickle file format

        :param infile: the /path/file.npy
    """

    with open(infile, 'rb') as handle:
        b = pickle.load(handle)

    return b
