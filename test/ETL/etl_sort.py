#!/usr/bin/python
# -*- coding: utf-8 -*-
from functions.etl.sort import SortOperation
from pycompss.api.api import compss_wait_on

import pandas as pd

import uuid


def my_random_string(string_length):
    """Returns a random string of length string_length."""
    random = str(uuid.uuid4()) # Convert UUID format to a Python string.
    return random[0:string_length] # Return the random string.


def generate_input(numFrag, size):
    from functions.data.data_functions import Partitionize
    import numpy.random as np

    i1 = np.randint(1000, size=size)
    i2 = np.randint(10, size=size)
    strings = [my_random_string(l) for l in i2]
    data = {'CRSDepTime': i1,
            'strings': strings}

    col_all = ['CRSDepTime', 'strings']
    data = pd.DataFrame(data)
    info = {'CRSDepTime': data.sort_values(['CRSDepTime'])['CRSDepTime'].values,
            'strings': data.sort_values(['strings'])['strings'].values,
            'all': data.sort_values(col_all)[col_all].values,
            }
    data0 = Partitionize(data, numFrag)

    return data0, info


if __name__ == '__main__':
    """Test Sort function."""
    numFrag = 8

    data0, info = generate_input(numFrag, 709)
    nums_sort = info['CRSDepTime']

    settings = dict()
    settings['columns'] = ['CRSDepTime']
    settings['ascending'] = [True]
    data1 = SortOperation().transform(data0, settings, numFrag)

    data1 = compss_wait_on(data1)
    data1 = pd.concat(data1, ignore_index=True, sort=False)
    nums = data1['CRSDepTime'].values

    print not any(nums_sort != nums)

