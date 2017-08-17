#!/usr/bin/python
# -*- coding: utf-8 -*-


import AddColumns
import Aggregation
import Difference
import Drop
import RemoveDuplicated
import Filter
import Intersect
import Join
import Sample
import Select
import Sort
import Transform
import Union



if __name__ == "__main__":
    pass
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
    #print Join(data,data2,1,1)
