#!/usr/bin/python
# -*- coding: utf-8 -*-


from pycompss.api.api import compss_wait_on
from functions.data.read_data import ReadOperationHDFS
from functions.data.balancer import WorkloadBalancerOperation
from random import randint

if __name__ == '__main__':
    """Test Balancer function."""

    numFrag = 4
    settings = dict()
    settings['port'] = 9000
    settings['host'] = 'localhost'
    filename = '/flights.csv'
    settings['header'] = True
    settings['separator'] = ','

    data0 = ReadOperationHDFS().transform(filename, settings, numFrag)

    data0 = compss_wait_on(data0)
    start = 0
    for f in range(numFrag):
        i = randint(0, 220)
        data0[f] = data0[f][0:i]
        print len(data0[f])

    data = WorkloadBalancerOperation().transform(data0, False, numFrag)

    data = compss_wait_on(data)

    for d in data:
        print len(d)

