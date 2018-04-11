#!/usr/bin/python
# -*- coding: utf-8 -*-

from pycompss.api.api import compss_wait_on
from functions.data.ReadData import ReadCSVFromHDFSOperation
from functions.etl.Sort import SortOperation
import pandas as pd

if __name__ == '__main__':
    """Test Sort function."""
    numFrag = 4
    settings = dict()
    settings['port'] = 9000
    settings['host'] = 'localhost'
    settings['path'] = '/flights.csv'
    settings['separator'] = ','

    data0 = ReadCSVFromHDFSOperation(settings, numFrag)
    data0 = compss_wait_on(data0)
    nums_sort = pd.concat(data0, ignore_index=True)['CRSDepTime'].values
    nums_sort = sorted(nums_sort)

    settings = dict()
    settings['columns'] = ['CRSDepTime']
    settings['ascending'] = [True]
    settings['algorithm'] = 'bitonic'  # 'odd-even' or 'bitonic'
    data1 = SortOperation().transform(data0, settings, numFrag)

    data1 = compss_wait_on(data1)

    nums = pd.concat(data1, ignore_index=True)['CRSDepTime'].values
    print nums
    #for i, z in zip(nums_sort, nums):
    #    print i, z

    print all(nums_sort == nums)
