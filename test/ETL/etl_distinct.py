#!/usr/bin/python
# -*- coding: utf-8 -*-

from pycompss.api.api import compss_wait_on
from functions.etl.distinct import DistinctOperation


def generate_input(numFrag):
    from functions.data.data_functions import Partitionize
    import numpy.random as np
    import pandas as pd

    i1 = np.randint(1000, size=1000)
    i2 = np.randint(600, size=1000)
    data = {'name': i1,
            'year': i2}

    data = pd.DataFrame(data)
    info = {'name': len(data['name'].unique()),
            'year': len(data['year'].unique()),
            'all': len(data.drop_duplicates(subset=['name', 'year']))
            }
    data0 = Partitionize(data, numFrag)
    print info
    return data0, info


if __name__ == '__main__':
    """Test Distinct function."""
    numFrag = 4

    data0, info = generate_input(numFrag)

    columns = ['year']
    data1 = DistinctOperation().transform(data0, columns, numFrag)
    data1 = compss_wait_on(data1)
    total = sum([len(d) for d in data1])
    print total == info['year']

    columns = ['name']
    data2 = DistinctOperation().transform(data0, columns, numFrag)
    data2 = compss_wait_on(data2)
    total = sum([len(d) for d in data2])
    print total == info['name']

    columns = ['name', 'year']
    data3 = DistinctOperation().transform(data0, columns, numFrag)
    data3 = compss_wait_on(data3)
    total = sum([len(d) for d in data3])
    print total == info['all']


