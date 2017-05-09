#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys

sys.path.insert(0, '/home/lucasmsp/workspace/BigSea/Compss-Python/functions/data')
from data_functions     import *
from linearRegression   import *

if __name__ == "__main__":
    from numpy import arange
    from numpy.random import randint
    from pylab import scatter, show, plot, savefig
    data = [[[1,2,3],[4,5,6]], [[1,2,3],[4,5,6]]]
    #data = [[list(randint(100, size=1000)) for _ in range(10)] for _ in range(2)]
    line = fit(data[0], data[1])
    print [line(x) for x in arange(0.0,100.0,1.0)]
    datax = [item for sublist in data[0] for item in sublist]
    datay = [item for sublist in data[1] for item in sublist]
    scatter(datax, datay, marker='x')
    plot([line(x) for x in arange(0.0, 10.0, 0.1)], arange(0.0, 10.0, 0.1))
    savefig('lrd.png')
