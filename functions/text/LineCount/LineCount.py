#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__ = "Lucas Miguel S Ponce"
__email__  = "lucasmsp@gmail.com"

from pycompss.api.task import task
from pycompss.api.parameter    import *
from pycompss.functions.reduce import mergeReduce
from pycompss.functions.data   import chunks

import string
import re
import unicodedata
import sys
reload(sys)
sys.setdefaultencoding("utf-8")


@task(returns=list, filename= FILE_IN)
def computeLineCount(filename):
    count = [0,0]
    for i in open(filename,"r"):
        count[0]+=1

    return count


def LinedCount(filename,numFrag):

    from pycompss.api.api import compss_wait_on
    partialResult = [computeLineCount("%s_%02d" % (filename,i)) for i in xrange(numFrag)]
    partialResult = compss_wait_on(partialResult)

    count =0
    for i in partialResult:
        count+=i[0]

    print count

if __name__ == "__main__":


        LinedCount("higgs-train-1m.csv",4)
