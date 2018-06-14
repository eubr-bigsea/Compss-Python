#!/usr/bin/python
# -*- coding: utf-8 -*-

from pycompss.api.api import compss_wait_on
from functions.etl.split import SplitOperation
import pandas as pd


def generate_input(numFrag):
    import numpy.random as np

    total = 0
    data = [[] for _ in range(numFrag)]
    for i in range(numFrag):
        size1 = np.randint(300, size=1)
        i1 = np.randint(1000, size=size1)
        content = {'col0': i1}
        data[i] = pd.DataFrame(content)
        print 'Fragment #{}: {}'.format(i, len(data[i]))
        total += len(data[i])

    return data, total


if __name__ == '__main__':
    """Test split function."""
    numFrag = 4

    data0, total = generate_input(numFrag)

    settings = dict()
    settings['percentage'] = 0.25
    data1, data2 = SplitOperation().transform(data0, settings, numFrag)

    data1 = compss_wait_on(data1)
    data2 = compss_wait_on(data2)

    data1 = pd.concat(data1, sort=False)
    data2 = pd.concat(data2, sort=False)

    count1 = len(data1)
    count2 = len(data2)
    print (count1 + count2) == total

    print "total:", total
    print "split1:", count1
    print "split2:", count2

