#!/usr/bin/python
# -*- coding: utf-8 -*-


from pycompss.api.task import task
from pycompss.api.parameter import *
from pycompss.functions.reduce import mergeReduce
from pycompss.functions.data import chunks
from pycompss.api.api import compss_wait_on

import numpy as np
import pandas as pd
import math

#------------------------------------------------------------------------------
# Merge two tables side to side

def AddColumns(list1,list2,numFrag):
    """
        Function which add new columns in the np.array.
        The output is already merged.

        :param list1: A np.array with already splited in numFrags.
        :param list2: A np.array with the columns to be added already
                    splitted in numFrags.
        :return: Returns a new np.array.
    """


    result = [AddColumns_part(list1[f], list2[f]) for f in range(numFrag)]
    #result = mergeReduce(Union_part,result)
    #result = compss_wait_on(result)
    return result

@task(returns=list)
def AddColumns_part(a,b):
    print "\nAddColumns_part\n---\n{}\n---\n{}\n---\n".format(a,b)
    if len(a)>0:
        if len(b)>0:
            return pd.concat([a, b], axis=1) #np.concatenate((a, b), axis=1)
        else:
            return a
    else:
        return b

#------------------------------------------------------------------------------


#-------------------------------------------------------------------------------
# Drop columns

def Drop(data, columns,numFrag):
    """
        Function which remove one or more columns in a np.array.
        The output remains splited.

        :param data: A np.array with already splited in numFrags.
        :param columns: A list with the indexs which will be removed.
        :return: Returns a np.array without the columns choosed.
    """

    from pycompss.api.api import compss_wait_on

    data_result = [Drop_part(data[f], columns) for f in range(len(data))]
    #data_result = compss_wait_on(data_result)

    return data_result

@task(returns=list)
def Drop_part(list1,columns):
    print "\nDrop_part\n"
    return  list1.drop(columns, axis=1) #np.delete(list1, columns, axis=1)

#-------------------------------------------------------------------------------



#-------------------------------------------------------------------------------
# Select/Projection columns in a array

def Select(data,columns,numFrag):
    """
        Function which do a Projection with the columns choosed.
        The output remains splitted.

        :param data: A np.array with already splited in numFrags.
        :param columns: A list with the indexs which will be selected.
        :return: Returns a np.array with only the columns choosed.
    """

    from pycompss.api.api import compss_wait_on
    data = compss_wait_on(data)
    data_result = [Select_part(data[f],columns) for f in range(numFrag)]
    #data_result = mergeReduce(Union_part,data_result)
    #data_result = compss_wait_on(data_result)

    return data_result

def Select_part(list1,fields):
    return list1[fields]
    #return np.array(list1)[:,fields]
#-------------------------------------------------------------------------------


#-------------------------------------------------------------------------------
# Union of the datasets

def Union(data1, data2,numFrag):
    """
        Function which do a union between two np.arrays.
        The output remains splitted.

        :param data1: A np.array with already splited in numFrags.
        :param data2: Other np.array with already splited in numFrags.
        :return: Returns a new np.arrays.
    """
    data1 = compss_wait_on(data1, to_write=False)
    data2 = compss_wait_on(data2, to_write=False)


    data_result = [Union_part(data1[f], data2[f]) for f in range(numFrag)]
    #data_result = mergeReduce(Union_part,data_result)
    #data_result = compss_wait_on(data_result)

    return data_result


@task(returns=list)
def Union_part(list1,list2):
    print "\nUnion_part\n---\n{}\n---\n{}\n---\n".format(list1,list2)

    if len(list1) == 0:
        result = list2
    elif len(list2) == 0:
        result = list1
    else:
        result = pd.concat([list1,list2], ignore_index=True)
    return  result
#-------------------------------------------------------------------------------


#-------------------------------------------------------------------------------
# Remove duplicate rows in a array

def DropDuplicates(data1,keys):
    """
        Function which remove duplicates elements (distinct elements) in a np.array.
        The output is already merged.

        :param name: A np.array with already splited in numFrags
        :return: Returns a new np.array
    """
    from pycompss.api.api import compss_wait_on

    data_merged  = mergeReduce(Union_part, data1)
    data_result  = DropDuplicates_merge(data_merged,keys)

    #data_result  =
    #data_result  = compss_wait_on(data_result)

    return data_result

@task(returns=list)
def DropDuplicates_merge(list1,keys):
    #  combine them excluding any duplicates
    # part =  np.concatenate((part1,part2), axis=0)
    # x = np.random.rand(part.shape[1])
    # y = part.dot(x)
    # unique, index = np.unique(y, return_index=True)
    print "\nDropDuplicates_merge\n---\n{}\n---\n{}\n---\n".format(list1,keys)

    if len(list1)>0:
            #part = pd.concat([list1[0],list2[0]])
        part = list1.drop_duplicates(keys, keep='last')
        print part
        return part
    else:
        return list1


#-------------------------------------------------------------------------------


#-------------------------------------------------------------------------------
# Intersect

def Intersect(data1,data2,numFrag):
    """
        Function which returns a new set with elements that are common to all sets.
        The output is already merged.

        :param data1: A np.array with already splited in numFrags
        :param data2: A np.array with already splited in numFrags
        :return: Returns a  new np.array
    """


    from pycompss.api.api import compss_wait_on

    data_result = [[] for i in range(numFrag)]

    for i in xrange(numFrag):
        data_partial = [ Intersect_part(data1[i],data2[j]) for j in xrange(numFrag) ]
        data_result[i] =  mergeReduce(Union_part,data_partial)
    #data_result = compss_wait_on(data_result)

    return data_result


@task(returns=list)
def Intersect_part(list1,list2):
    print "\nIntersect_part\n---\n{}\n---\n{}\n---\n".format(list1,list2)
    if len(list1) == 0 or len(list2) == 0:
        result = []
    else:
        result = list1.merge(list2)
        print result

    return  result
#-------------------------------------------------------------------------------



#-------------------------------------------------------------------------------
#   Difference

def Difference(data1,data2,numFrag):
    """
        Function which returns a new set with containing rows in the first frame
        but not in the second one.
        The output is already merged.

        :param data1: A np.array with already splited in numFrags
        :param data2: A np.array with already splited in numFrags
        :return: Returns a  new np.array
    """
    from pycompss.api.api import compss_wait_on

    data_result = [[] for i in range(len(data1))]

    for f1 in range(len(data1)):
        data_partial      = [ Difference_part(data1[f1], data2[f2]) for f2 in range(numFrag) ]
        data_result[f1]  = mergeReduce(Intersect_part, data_partial)

    #data_result  = mergeReduce(Union_part,data_partial)
    #data_result  = compss_wait_on(data_result)

    return data_result

@task(returns=list)
def Difference_part(df1,df2):
    # a1_rows = a1.view([('', a1.dtype)] * a1.shape[1])
    # a2_rows = a2.view([('', a2.dtype)] * a2.shape[1])
    # result = np.setdiff1d(a1_rows, a2_rows).view(a1.dtype).reshape(-1, a1.shape[1])
    if len(df1) > 0:
        if len(df2) > 0:
            names = df1.columns
            ds1 = set([ tuple(line) for line in df1.values.tolist()])
            ds2 = set([ tuple(line) for line in df2.values.tolist()])
            result = pd.DataFrame(list(ds1.difference(ds2)))
            result.columns = names
            print result
            return result
        else:
            return df2
    else:
        return []

#-------------------------------------------------------------------------------
#   Join

@task(returns=list)
def InnerJoin(data1,data2,id1,id2):

    print data1
    print "----"
    print data2

    if len(data1)>0 and len(data2)>0:
            df_partial = pd.merge(data1,data2, how='inner', left_on=id1, right_on=id2)
            print df_partial
            return df_partial
    else:
            return []

    return df_partial

@task(returns=list)
def LeftJoin(data1,data2,id1,id2):

    print data1
    print "----"
    print data2

    if len(data1)>0 :
        if len(data2)>0 :

            df_partial = pd.merge(data1,data2, how='left', left_on=id1, right_on=id2,indicator=True)
            idx = list(set(id1+id2))
            df_partial.set_index(id1)
            print df_partial
            return df_partial
        else:
            df_partial = data1
            print df_partial
            return df_partial
    else:
        df_partial = data2
        print df_partial
        return df_partial


@task(returns=list)
def RightJoin(data1,data2,id1,id2):
    L = len(data2)

    if L ==0:
        return [[],[]]
    if len(data1) == 0:
        return [[],[]]

    C = len(data1[0]) + len(data2[0]) - len(id1)
    size = len(data1[0]) - len(id1)

    #print "L :{} | C (A+B) : {} |  sub(A-B): {}".format(L,C,size)

    b = np.zeros((L,C ))
    b[:,size:] = data2
    log = np.zeros((L,1 ))

    print b

    for i in range(len(data2)):
        for j in range(len(data1)):
            found=True

            for i1,j2 in  zip(id1,id2):
                if data1[i][i1] != data2[j][j2]:
                    found=False
            if found:
                sub = np.delete(data1[j], id2, None)

                b[i][:size] =  sub
                log[i]=1


    print [b,log]

    return [b,log]



@task(returns=list)
def mergeInnerJoin(data1,data2):

    if len(data1)>0:
        if len(data2)>0:
            return pd.concat([data1,data2])
        else:
            return data1
    else:
        return data2

@task(returns=list)
def mergeLeftJoin(data1,data2,id1,id2):

    print data1
    print "---"
    print data2

    if len(data1)>0:
        if len(data2)>0:
            data = data1.set_index(id1).merge(data2.set_index(id2))
            print  data
            return data
        else:
            return data1
            #log = data2[1]
    else:
        return data2

@task(returns=list)
def mergeRightJoin(data1,data2,id1,id2):

    print data1
    print id1
    print "---"
    print data2
    print id2

    if len(data1)>0:
        if len(data2)>0:
            data = data2.set_index(id2).merge(data1.set_index(id1))
            print  data
            return data
        else:
            return data1
            #log = data2[1]
    else:
        return data2

def Join (data1,data2,id1,id2,params,numFrag):
    result = [[] for i in range(numFrag)]

    if params['option'] == "inner":
        for i in range(numFrag):
            partial_join    = [InnerJoin(data1[i],data2[j],id1,id2) for j in range(numFrag)]
            result[i]  = mergeReduce(mergeInnerJoin,partial_join)
        return result
            #merged_join[i]  = merged_join[i][0]       ------------- TEm como enviar 2 ou mais ##############

    elif params['option'] == "left":
        partial_m = [[] for i in range(numFrag)]
        for i in range(numFrag):
            partial_join    = [InnerJoin(data1[i],data2[j],id1,id2) for j in range(numFrag)]
            partial_m[i]    = mergeReduce(mergeInnerJoin,partial_join)
            result[i]       = mergeLeftJoin(partial_m[i],data1[i],id2,id1)
        return result
    elif params['option'] == "right":
        partial_m = [[] for i in range(numFrag)]
        for i in range(numFrag):
            partial_join    = [InnerJoin(data1[i],data2[j],id1,id2) for j in range(numFrag)]
            partial_m[i]    = mergeReduce(mergeInnerJoin,partial_join)
            result[i]       = mergeRightJoin(partial_m[i],data2[i],id1,id2)
        return result






def Sort(data, ids,order,numFrag):
    partial_m = [[] for i in range(numFrag)]
    for i in range(numFrag):
        partial_m[i] = sort_partial(data[i],ids,order)

    return partial_m

def sort_partial(df,ids,order):
    return df.sort(ids, ascending=order)



def FilterC(data,numFrag):
    partial_m = [[] for i in range(numFrag)]
    for i in range(numFrag):
        partial_m[i] = filter_partial(data[i])

    return partial_m

#FICARA no master
@task(returns=list)
def filter_partial(data):
    return data[data['id1'] > 3]



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

    size_split1 = int(math.ceil(total[0]*percentage))

    np.random.seed(seed)
    ids1 = sorted(np.random.choice(total[0], size_split1, replace=False))
    ids2 = [i for i in range(size_split1) if i not in ids1]

    ids = [ids1,ids2]
    # list_ids = [[] for i in range(numFrag)]
    # frag = 0
    # maxIdFrag = total[1][frag]
    # oldmax = 0
    # for i in ids:
    #     while i >= maxIdFrag:
    #         frag+=1
    #         oldmax = maxIdFrag
    #         maxIdFrag+= total[1][frag]
    #     list_ids[frag].append(i-oldmax)

    #print "Total: {} |\nsize_split1: {} |\nids: {} |\nlist_ids:{}".format(total,size_split1,ids,list_ids)

    return ids

@task(returns=list)
def GetSplit1(data,indexes_split1):
    split1 = []
    print "DEGUG: GetSplit1"

    if len(data)>0:
        print data
        print data.index
        print "List of index: %s" % indexes_split1

        #df.loc[~df.index.isin(t)]
        split1 = data.loc[data.index.isin(indexes_split1)]


    #     if
    # pos= 0
    # if len(indexes_split1)>0:
    #     for i  in range(len(data)):
    #         if i == indexes_split1[pos]:
    #             split1.append(data[i])
    #             if pos < (len(indexes_split1)-1):
    #                 pos+=1

    print split1
    return split1

@task(returns=list)
def GetSplit2(data,indexes_split2):
    print "DEGUG: GetSplit2"
    split2 = []
    pos= 0

    if len(data)>0:
        print data
        print data.index
        print "List of index: %s" % indexes_split2

        split2 =data.loc[data.index.isin(indexes_split2)]


    print split2
    return split2


def Split(data,settings,numFrag):
    percentage = settings['percentage']
    seed = settings['seed']
    from pycompss.api.api import compss_wait_on
    data = compss_wait_on(data,to_write = False)
    partial_counts = [CountRecord(data[i]) for i in range(numFrag)]
    total = mergeReduce(mergeCount,partial_counts)
    indexes = DefineSplit(total,percentage,seed,numFrag)
    indexes = compss_wait_on(indexes,to_write = False)
    splits1 = [GetSplit1(data[i],indexes[0]) for i in range(numFrag)]
    splits2 = [GetSplit2(data[i],indexes[1]) for i in range(numFrag)]
    return  [splits1,splits2]



#-------------------------------------------------------------------------------
#   Sample

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

    print "Total: {} |\nsize: {} |\nids: {} |\nlist_ids:{}".format(total,value,ids,list_ids)

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
    indexes_split1 = [[] for i in range(numFrag)]
    if params["type"] == 'percent':
        percentage  = params['value']
        seed        = params['seed']

        partial_counts  = [CountRecord(data[i]) for i in range(numFrag)] #Remove in the future
        total           = mergeReduce(mergeCount,partial_counts)
        indexes         = DefineSample(total,percentage,seed,numFrag)
        indexes = compss_wait_on(indexes,to_write = False)
        sample = [GetSample(data[i],indexes[i]) for i in range(numFrag)]
        return sample
    elif params["type"] == 'value':
        value = params['value']
        seed = params['seed']
        partial_counts = [CountRecord(data[i]) for i in range(numFrag)]
        total = mergeReduce(mergeCount,partial_counts)
        indexes_split1 = DefineNSample(total,value,seed,numFrag)
        indexes_split1 = compss_wait_on(indexes_split1,to_write = False)
        splits1 = [GetSample(data[i],indexes_split1[i]) for i in range(numFrag)]
        return splits1
    elif params['type'] == 'head':
        head = params['value']
        partial_counts = [CountRecord(data[i]) for i in range(numFrag)]
        total = mergeReduce(mergeCount,partial_counts)
        total = compss_wait_on(total,to_write = False)
        sample = [GetHeadSample(data[i], total,i,head) for i in range(numFrag)]
        return sample


@task(returns=list)
def DefineSample(total,percentage,seed,numFrag):

    size = int(math.ceil(total[0]*percentage))

    np.random.seed(seed)
    ids = sorted(np.random.choice(total[0], size, replace=False))


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

    print "Total: {} |\nsize: {} |\nids: {} |\nlist_ids:{}".format(total,size,ids,list_ids)

    return list_ids




@task(returns=list)
def GetSample(data,indexes):
    sample = []
    print "DEGUG: GetSample"

    if len(data)>0:
        print data
        print data.index
        print "List of index: %s" % indexes

        data = data.reset_index(drop=True)
        sample = data.loc[data.index.isin(indexes)]



    print sample
    return sample

if __name__ == "__main__":

    # data = np.array([[i,6,3] for i in range(10)] + [[i,6,3] for i in range(5, 17)] )
    # data2 = np.array([[i,6,3] for i in range(11, 15) ])
    # data3 = np.array([[i,-100,-100] for i in range(11, 15) ])
    # numFrag = 4
    # data = [d for d in chunks(data, len(data)/numFrag)]
    # data2 = [d for d in chunks(data2, len(data2)/numFrag)]
    # data3 = [d for d in chunks(data3, len(data3)/numFrag)]
    # print "{} --> {}".format(len(data),data)

    ##-----------------------------
    #print "Drop Example:" # OK
    #print Drop(data,[1,2])

    ##-----------------------------
    #print "Projection/Select Example:" # OK
    #print Select(data,[1])

    ##-----------------------------
    #print "AddColumns Example:"  # OK
    #print AddColumns(data,data)

    ##-----------------------------
    #print "Union Example:"  # OK
    #print Union(data,data3)

    ##-----------------------------
    #print "Intersection Example:" # OK
    #print Intersect(data,data2)

    ##-----------------------------
    #print "Difference Example:"  # OK
    #print Difference(data,data2)

    ##-----------------------------
    #print "DropDuplicates Example:" # OK
    #print  DropDuplicates(data)

    ##-----------------------------
    #print "Join Example:" #
    print Join(data,data2,1,1)
