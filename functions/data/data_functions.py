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

#@task(returns=list)
def Partitionize(data,numFrag):
    """
        Partitionize:

        Method to split the data in numFrags parts. This method simplifies
        the use of chunks.

        :param data: The np.array or list to do the split.
        :param numFrags: A number of partitions
        :return The array splitted.
    """

    PartitionSize = int(math.ceil(float(len(data))/numFrag))
    Ds = [d for d in chunks(data, PartitionSize )]
    partitions = [pd.DataFrame() for _ in range(numFrag)]
    for d in range(len(Ds)):
        partitions[d] = Ds[d]

    return partitions




#------------------------------------------------------------------------------
# Save Methods



@task(filename = FILE_OUT)
def SaveToFile(filename,data,mode,header):
    """
        SaveToFile (CSV):

        Method used to save an array into a file.

        :param filename: The name used in the output.
        :param data: The np.array which you want to save.
        :param mode: append, overwrite, ignore or error

    """
    import os.path


    if mode is 'append':
        mode = 'a'
    elif mode is 'ignore':
        if os.path.exists(filename):
            return None
    elif mode is 'error':
        if os.path.exists(filename):
            return None    # !   TO DO: RAISE SOME ERROR
    else:
        mode = 'w'

    print data
    if len(data)==0:
        data = pd.DataFrame()
    if header:
        data.to_csv(filename,sep=',',mode=mode, header=True,index=False)
    else:
        data.to_csv(filename,sep=',',mode=mode, header=False,index=False)

    return None


def SaveToPickle(outfile,data):
    """
        Save an array to a serizable Pickle file format

        :param outfile: the /path/file.npy
        :param data: the data to save
    """
    with open(outfile, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

def SaveToNumpy(outfile,data):
    """
        Save an array to a binary file in NumPy .npy format

        :param outfile: the /path/file.npy
    """
    np.save(outfile, data)
    return None

#------------------------------------------------------------------------------
# Load Methods


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

#@task(returns=list)
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



def VectorAssemble(data, col):
    #Nao posso ter multidimensoes por limitacao do numpy
    #col0 sempre será label
    # label = []
    # features = []
    # for row in data:
    #     feature = []
    #     for i in range(len(row)):
    #         if i == col:
    #             label.append(row[i])
    #         else:
    #             feature.append(row[i])
    #     features.append(feature)
    #
    # label = np.asarray(label)
    # features = np.asarray(features)
    # vector = [label,features]
    #return vector = np.asarray(vector)
    order = [col]
    for i in range(len(data[0])):
        if i !=col:
            order.append(i)

    i = np.argsort(order)
    return data[:,i]
