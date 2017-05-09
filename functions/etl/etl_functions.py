#!/usr/bin/python
# -*- coding: utf-8 -*-


from pycompss.api.task import task
from pycompss.api.parameter import *
from pycompss.functions.reduce import mergeReduce
from pycompss.functions.data import chunks

import numpy as np


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

    from pycompss.api.api import compss_wait_on
    result = [AddColumns_part(list1[f], list2[f]) for f in range(numFrag)]
    result = mergeReduce(Union_part,result)
    #result = compss_wait_on(result)
    return result

@task(returns=list)
def AddColumns_part(a,b):
    return np.concatenate((a, b), axis=1)
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
    return  np.delete(list1, columns, axis=1)

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

    data_result = [Select_part(data[f],columns) for f in range(numFrag)]
    data_result = mergeReduce(Union_part,data_result)
    #data_result = compss_wait_on(data_result)

    return data_result

def Select_part(list1,fields):
    return np.array(list1)[:,fields]
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

    from pycompss.api.api import compss_wait_on

    data_result = [Union_part(data1[f], data2[f]) for f in range(numFrag)]
    data_result = mergeReduce(Union_part,data_result)
    #data_result = compss_wait_on(data_result)

    return data_result


@task(returns=list)
def Union_part(list1,list2):
    if len(list1) == 0:
        result = list2
    elif len(list2) == 0:
        result = list1
    else:
        result = np.concatenate((list1,list2), axis=0)
    return  result
#-------------------------------------------------------------------------------


#-------------------------------------------------------------------------------
# Remove duplicate rows in a array

def DropDuplicates(data1):
    """
        Function which remove duplicates elements (distinct elements) in a np.array.
        The output is already merged.

        :param name: A np.array with already splited in numFrags
        :return: Returns a new np.array
    """
    from pycompss.api.api import compss_wait_on

    data_result  = mergeReduce(DropDuplicates_merge, data1)
    #data_result  = compss_wait_on(data_result)

    return data_result

@task(returns=list)
def DropDuplicates_merge(part1,part2):
    #  combine them excluding any duplicates
    part =  np.concatenate((part1,part2), axis=0)
    x = np.random.rand(part.shape[1])
    y = part.dot(x)
    unique, index = np.unique(y, return_index=True)

    return  part[index] #np.unique(np.concatenate((part1,part2),axis=0)) #list(set(part1 + part2)) # [x for x in list2 if x not in  list1]


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

    data_partial = [ Intersect_part(data1[i],data2[j])
                    for i in xrange(numFrag)  for j in xrange(numFrag) ]
    data_result =  mergeReduce(Union_part,data_partial)
    #data_result = compss_wait_on(data_result)

    return data_result


@task(returns=list)
def Intersect_part(list1,list2):
    print list1
    print list2
    if (len(list1)) == 0:
        result = []
    else:
        nrows, ncols = list1.shape
        dtype={'names':['f{}'.format(i) for i in range(ncols)],
               'formats':ncols * [list1.dtype]}

        result = np.intersect1d(list1.view(dtype), list2.view(dtype))
        new = []
        for e in result:
            row = []
            for i in e:
                row.append(i)
            new.append(row)
        result = np.array(new)
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

    data_partial = [[] for i in range(len(data1))]

    for f1 in range(len(data1)):
        data_partial[f1] = [ Difference_part(data1[f1], data2[f2]) for f2 in range(numFrag) ]
        data_partial[f1]  = mergeReduce(Intersect_part, data_partial[f1])

    data_result  = mergeReduce(Union_part,data_partial)
    #data_result  = compss_wait_on(data_result)

    return data_result

@task(returns=list)
def Difference_part(a1,a2):
    a1_rows = a1.view([('', a1.dtype)] * a1.shape[1])
    a2_rows = a2.view([('', a2.dtype)] * a2.shape[1])
    result = np.setdiff1d(a1_rows, a2_rows).view(a1.dtype).reshape(-1, a1.shape[1])

    return result
#-------------------------------------------------------------------------------
#   Join

@task(returns=list)
def InnerJoin(data1,data2,id1,id2):
    L = len(data1)

    if L ==0:
        return [[],[]]
    if len(data2) == 0:
        return [[],[]]

    C = len(data1[0]) + len(data2[0]) - len(id1)
    size = len(data2[0]) - len(id1)

    #print "L :{} | C (A+B) : {} |  sub(A-B): {}".format(L,C,size)

    b = np.zeros((L,C ))
    b[:,:-size] = data1
    log = np.zeros((L,1 ))

    for i in range(len(data1)):
        for j in range(len(data2)):
            found=True

            for i1,j2 in  zip(id1,id2):
                if data1[i][i1] != data2[j][j2]:
                    found=False
            if found:
                sub = np.delete(data2[j], id2, None)

                b[i][-size:] =  sub
                log[i]=1


    indices = [i for (i,v) in enumerate(log) if v==0]
    b = np.delete(b, indices, 0)
    log = np.delete(log, indices, 0)

    return [b,log]

@task(returns=list)
def LeftJoin(data1,data2,id1,id2):

    L = len(data1)

    if L ==0:
        return [[],[]]
    if len(data2) == 0:
        return [[],[]]

    C = len(data1[0]) + len(data2[0]) - len(id1)
    size = len(data2[0]) - len(id1)

    #print "L :{} | C (A+B) : {} |  sub(A-B): {}".format(L,C,size)

    b = np.zeros((L,C ))
    b[:,:-size] = data1
    log = np.zeros((L,1 ))

    for i in range(len(data1)):
        for j in range(len(data2)):
            found=True

            for i1,j2 in  zip(id1,id2):
                if data1[i][i1] != data2[j][j2]:
                    found=False
            if found:
                sub = np.delete(data2[j], id2, None)

                b[i][-size:] =  sub
                log[i]=1


    #print [b,log]

    return [b,log]

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
def mergeOuterJoin(data1,data2):

    data = data1[0]
    log = data1[1]

    print data1
    print data2

    if len(log)>0:
        if len(data2[1])>0:
            for i in range(len(log)):
                if data2[1][i] == 1:
                    data[i] = data2[0][i]
                    log[i] = 1
    else:
        data = data2[0]
        log = data2[1]

    result = [np.array(data), log ]

    print result

    return result


@task(returns=list)
def mergeInnerJoin(data1,data2):

    if len(data2[1])>0:
        if len(data1[1])>0:
            data = np.concatenate((data1[0], data2[0]), axis=0)
            log  = np.concatenate((data1[1], data2[1]), axis=0)
        else:
            data = data2[0]
            log = data2[1]
    elif len(data1[1])>0:
        data = data1[0]
        log = data1[1]
    else:
        return [[],[]]

    result = [np.array(data), log ]

    return result


def Join (data1,data2,id1,id2,params,numFrag):

    merged_join = [[] for i in range(numFrag)]
    if params['option'] == "inner":
        for i in range(numFrag):
            partial_join    = [InnerJoin(data1[i],data2[j],id1,id2) for j in range(numFrag)]
            merged_join[i]  = mergeReduce(mergeInnerJoin,partial_join)
            #merged_join[i]  = merged_join[i][0]       ------------- TEm como enviar 2 ou mais ##############

    elif params['option'] == "left":
        for i in range(numFrag):
            partial_join    = [LeftJoin(data1[i],data2[j],id1,id2) for j in range(numFrag)]
            merged_join[i]  = mergeReduce(mergeOuterJoin,partial_join)
    else:
        #TO DO
        for i in range(numFrag):
            partial_join    = [RightJoin(data1[i],data2[j],id1,id2) for j in range(numFrag)]
            merged_join[i]  = mergeReduce(mergeOuterJoin,partial_join)

    return merged_join








if __name__ == "__main__":

    data = np.array([[i,6,3] for i in range(10)] + [[i,6,3] for i in range(5, 17)] )
    data2 = np.array([[i,6,3] for i in range(11, 15) ])
    data3 = np.array([[i,-100,-100] for i in range(11, 15) ])
    numFrag = 4
    data = [d for d in chunks(data, len(data)/numFrag)]
    data2 = [d for d in chunks(data2, len(data2)/numFrag)]
    data3 = [d for d in chunks(data3, len(data3)/numFrag)]
    print "{} --> {}".format(len(data),data)

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
