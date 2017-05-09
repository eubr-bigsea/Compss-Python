# -*- coding: utf-8 -*-
#!/usr/bin/env python


from pycompss.functions.data import chunks
from pycompss.api.task          import task
from pycompss.api.parameter     import *
from pycompss.functions.reduce import mergeReduce

import numpy as np
import math
import pickle

@task(returns=list)
def Partitionize(data,numFrag):
    """
        Partitionize:

        Method to split the data in numFrags parts. This method simplifies
        the use of chunks.

        :param data: The np.array or list to do the split.
        :param numFrags: A number of partitions
        :return The array splitted.
    """

            # #Wrong:
            # data = np.arange(299)
            # numFrag = 3
            # PartitionSize = len(data)/numFrag
            # ndata = [d for d in chunks(data, PartitionSize )]
            # print "Total Len ({}): ".format(len(ndata))
            # for d in ndata:
            #     print "Len ({})".format(len(d))
            #
            # #Right:
            # PartitionSize = int(math.ceil(float(len(data))/numFrag))
            # ndata = [d for d in chunks(data, PartitionSize )]
            # print "Total Len ({}): ".format(len(ndata))
            # for d in ndata:
            #     print "Len ({})".format(len(d))


    PartitionSize = int(math.ceil(float(len(data))/numFrag))
    data = [d for d in chunks(data, PartitionSize )]

    q = numFrag - len(data)
    r = [[] for i in range(q)]
    data = data + r
    return data


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
        f_handle = open(filename, mode='a+')
    elif mode is 'ignore':
        if os.path.exists(filename):
            return None
    elif mode is 'error':
        if os.path.exists(filename):
            return None    # !   TO DO: RAISE SOME ERROR
    else:
        f_handle = open(filename, mode='w')



    if header:
        title = ";".join([i for i in data.dtype.names])
        np.savetxt(f_handle,data, delimiter=';',header=title, fmt='%s')
    else:
        np.savetxt(f_handle,data, delimiter=';', fmt='%s')

    f_handle.close()
    return None



@task(returns=list)
def Union_part(list1,list2):
    if len(list1) == 0:
        result = list2
    elif len(list2) == 0:
        result = list1
    else:
        result = np.concatenate((list1,list2), axis=0)
    return  result


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

def ReadFromFile(filename,separator,header,infer):
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
            data = np.genfromtxt(filename,dtype='str', names = None,
                                     delimiter='\n',skip_header=1)
        else:
            data = np.genfromtxt(filename,dtype='str', names = None,
                                     delimiter='\n',skip_header=0,comments=None)

    #     #https://docs.scipy.org/doc/numpy/reference/generated/numpy.genfromtxt.html#numpy.genfromtxt
    #     if header:
    #         data = np.genfromtxt(filename, dtype=None, names = None,
    #                                 delimiter=separator,skip_header=1)
    #     else:
    #         data = np.genfromtxt(filename, dtype=None, names = None,
    #                                 delimiter=separator,skip_header=0,comments=None)
    elif infer == "FROM_VALUES":
        if header:
            data = np.loadtxt(filename, delimiter=separator, skiprows=1,comments=None)
        else:
            data = np.loadtxt(filename, delimiter=separator, skiprows=0,comments=None)

    return data



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

#-------------------------------------------------------------------------------
#   Split

@task(returns=list)
def CountRecord(data):
    size = len(data)
    return [size,[size]]

@task(returns=list)
def mergeCount(data1,data2):
    return [data1[0]+data2[0],np.concatenate((data1[1], data2[1]), axis=0)]


@task(returns=list)
def DefineSplit (total,percentage,seed,numFrag):

    size_split1 = math.ceil(total[0]*percentage)

    np.random.seed(seed)
    ids = sorted(np.random.choice(total[0], size_split1, replace=False))

    list_ids = [[] for i in range(numFrag)]
    frag = 0
    maxIdFrag = total[1][frag]
    oldmax = 0
    for i in ids:
        while i >= maxIdFrag:
            frag+=1
            oldmax = maxIdFrag
            maxIdFrag+= total[1][frag]
        list_ids[frag].append(i-oldmax)

    #print "Total: {} |\nsize_split1: {} |\nids: {} |\nlist_ids:{}".format(total,size_split1,ids,list_ids)

    return list_ids

@task(returns=list)
def GetSplit1(data,indexes_split1):
    split1 = []

    pos= 0
    if len(indexes_split1)>0:
        for i  in range(len(data)):
            if i == indexes_split1[pos]:
                split1.append(data[i])
                if pos < (len(indexes_split1)-1):
                    pos+=1


    return split1

@task(returns=list)
def GetSplit2(data,indexes_split1):
    split2 = []
    pos= 0
    if len(indexes_split1)>0:
        for i  in range(len(data)):
            if i == indexes_split1[pos]:
                if pos < (len(indexes_split1)-1):
                    pos+=1
            else:
                split2.append(data[i])
    else:
        split2 = data

    return split2


def Split(data,percentage,seed,numFrag):
    from pycompss.api.api import compss_wait_on
    partial_counts = [CountRecord(data[i]) for i in range(numFrag)]
    total = mergeReduce(mergeCount,partial_counts)
    indexes_split1 = DefineSplit(total,percentage,seed,numFrag)
    indexes_split1 = compss_wait_on(indexes_split1,to_write = False)
    splits1 = [GetSplit1(data[i],indexes_split1[i]) for i in range(numFrag)]
    splits2 = [GetSplit2(data[i],indexes_split1[i]) for i in range(numFrag)]
    return  [splits1,splits2]


#-------------------------------------------------------------------------------
#   Sample

def GetHeadSample(data, total,i,head):
    others_workers = sum([total[1][j] for j in range(i)])
    #print others
    head -= others_workers
    if head>0:
        return data[0:head]
    return []


@task(returns=list)
def DefineNSample (total,value,seed,numFrag):

    if total[0] < value:
        value = total[0]
    np.random.seed(seed)
    ids = sorted(np.random.choice(total[0], value, replace=False))

    list_ids = [[] for i in range(numFrag)]

    frag = 0
    maxIdFrag = total[1][frag]
    oldmax = 0
    for i in ids:

        while i >= maxIdFrag:
            frag+=1
            oldmax = maxIdFrag
            maxIdFrag+= total[1][frag]

        list_ids[frag].append(i-oldmax)

    print "Total: {} |\nsize_split1: {} |\nids: {} |\nlist_ids:{}".format(total,value,ids,list_ids)

    return list_ids

def Sample(data,params,numFrag):
    """
    Returns a sampled subset of this DataFrame.
    Parameters:
    - withReplacement -> can elements be sampled multiple times
                        (replaced when sampled out)
    - fraction -> fraction of the data frame to be sampled.
        without replacement: probability that each element is chosen;
            fraction must be [0, 1]
        with replacement: expected number of times each element is chosen;
            fraction must be >= 0
    - seed -> seed for random operation.
    """
    from pycompss.api.api import compss_wait_on
    if params["type"] == 'percent':
        percentage = params['percent']
        seed = params['seed']
        partial_counts = [CountRecord(data[i]) for i in range(numFrag)]
        total = mergeReduce(mergeCount,partial_counts)
        indexes_split1 = DefineSplit(total,percentage,seed,numFrag)
        indexes_split1 = compss_wait_on(indexes_split1,to_write = False)
        splits1 = [GetSplit1(data[i],indexes_split1[i]) for i in range(numFrag)]
        return splits1
    elif params["type"] == 'value':
        value = params['value']
        seed = params['seed']
        partial_counts = [CountRecord(data[i]) for i in range(numFrag)]
        total = mergeReduce(mergeCount,partial_counts)
        indexes_split1 = DefineNSample(total,value,seed,numFrag)
        indexes_split1 = compss_wait_on(indexes_split1,to_write = False)
        splits1 = [GetSplit1(data[i],indexes_split1[i]) for i in range(numFrag)]
        return splits1
    elif params['type'] == 'head':
        head = params['value']
        partial_counts = [CountRecord(data[i]) for i in range(numFrag)]
        total = mergeReduce(mergeCount,partial_counts)
        total = compss_wait_on(total,to_write = False)
        sample = [GetHeadSample(data[i], total,i,head) for i in range(numFrag)]
        return sample
