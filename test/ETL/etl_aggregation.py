#!/usr/bin/python
# -*- coding: utf-8 -*-

from pycompss.api.api import compss_wait_on
from functions.data.read_data import ReadOperationHDFS
from functions.etl.aggregation import AggregationOperation
import pandas as pd
import datetime


def generate_input(numFrag):
    import numpy.random as np

    data0 = [[] for _ in range(numFrag)]

    dt = datetime.datetime.now()
    dt2 = datetime.date.today()
    total = 0
    for i in range(numFrag):
        size1 = np.randint(250, size=1)
        content_i = np.randint(1000, size=size1)
        dates = [dt for _ in range(size1)]
        dates2 = [dt2 for _ in range(size1)]
        data = {'random': content_i, 'date': dates, 'date2': dates2}
        data0[i] = pd.DataFrame(data)
        total += size1[0]

    return data0, total


if __name__ == '__main__':
    """Test Aggregation function."""
    numFrag = 4

    data0, total = generate_input(numFrag)

    settings = dict()
    settings['columns'] = ["date", 'date2']
    settings['operation'] = {"date": [u'count']}
    settings['aliases'] = {"date": [u'COUNT']}

    data1 = AggregationOperation().transform(data0, settings, numFrag)
    data1 = compss_wait_on(data1)
    data1 = pd.concat(data1, sort=False)
    print data1.iloc[0]['COUNT'] == total

