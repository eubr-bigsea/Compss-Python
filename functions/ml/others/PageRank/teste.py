#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys
sys.path.insert(0, '/home/lucasmsp/workspace/BigSea/Compss-Python/functions/data')
from data_functions import *
from ListAdj import *
#from pagerank import *

inlink_file_name =  "/home/lucasmsp/workspace/BigSea/Compss-Python/functions/ml/others/PageRank/data"

inlink_file = open(inlink_file_name, 'r')
from pycompss.api.api import compss_wait_on

data = []
for line in inlink_file:
    row = [i.replace("\n","") for i in line.split(" ")]
    data.append(row)

from pycompss.functions.reduce import *

data1 = [[['MapR', 'Baidu'], ['MapR', 'Blogger']], [['Google', 'MapR'], ['Blogger', 'Baidu']], [['Blogger', 'Google'], ['Baidu', 'MapR']], []]
#data = [4,5,6,7,8,0,10,10,9]
#data = [[4,5,6,7,8],[0,10,10,9]]
print data1
#data1=data
numFrag = 4
from pycompss.functions.sort import *
#data1 = Partitionize(data,numFrag)
#data1 = compss_wait_on(data1)
#print data1
#merged = mergeReduce(sort,data1)


#merged= create_AdjList(data1,numFrag)

merged = Split(data1,3,3,4)

merged = compss_wait_on(merged)

"""
'MapR': ['Baidu', 'Blogger'], 'Baidu': ['MapR'], 'Google': ['MapR'], 'Blogger': ['Baidu', 'Google']
"""
print merged
