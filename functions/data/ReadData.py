# -*- coding: utf-8 -*-
#!/usr/bin/env python


#from pycompss.functions.data import chunks
from pycompss.api.task          import task
from pycompss.api.parameter     import *
from pycompss.functions.reduce import mergeReduce

import numpy as np
import math
import pickle
import pandas as pd


#------------------------------------------------------------------------------
# Load Methods



def ReadCSVOperation(filename,separator,header,infer,na_values):
    """
        ReadCSVOperation:

        Method used to load a pandas DataFrame from a csv file.

        :param filename:        The absolute path where the dataset is stored.
                                Each dataset is expected to be in a specific folder.
                                The folder will have the name of the dataset with the suffix "_folder".
                                The dataset is already divided into numFrags files, sorted lexicographically.
        :param separator:       The string used to separate values;
        :param header:          True if the first line is a header, otherwise is False;
        :param infer:
            - "NO":             Do not infer the data type of each column (will be string);
            - "FROM_VALUES":    Try to infer the data type of each column;
            - "FROM_LIMONERO":  !! NOT IMPLEMENTED YET!!
        :param na_values:       A list with the all nan caracteres to be considerated.
        :return                 A DataFrame splitted in a list with length N.

        Example:
        $ ls /var/workspace/dataset_folder
            dataset_00     dataset_02
            dataset_01     dataset_03
    """
    if infer == "FROM_LIMONERO":
        infer = "FROM_VALUES"

    import os, os.path
    data = []
    DIR = filename+"_folder"

    #BUG: COMPSs dont handle with a string "\n" as a parameter, using the tag "<new_line>" instead.
    data =  [ ReadCSV( "{}/{}".format(DIR,name),
                        separator,
                        header,
                        infer,
                        na_values) for name in sorted(os.listdir(DIR))
            ]

    return data

@task(returns=list, filename = FILE_IN)
def ReadCSV(filename,separator,header,infer,na_values):

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

    elif infer == "FROM_LIMONERO":
        df = pd.DataFrame([])

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
