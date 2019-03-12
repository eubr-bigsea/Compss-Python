#!/usr/bin/python
# -*- coding: utf-8 -*-

from pycompss.functions.data import chunks
import math
import pandas as pd


def parallelize(data, nfrag):
    """
       Method to split the data in nfrag parts. This method simplifies
       the use of chunks.

       :param data: The np.array or list to do the split.
       :param nfrag: A number of partitions
       :return The array splitted.

       :Note: the result may be unbalanced when the number of rows is too small
    """

    new_size = int(math.ceil(float(len(data))/nfrag))
    result = [d for d in chunks(data, new_size)]

    while len(result) < nfrag:
        result.append(pd.DataFrame(columns=result[0].columns))

    info = []
    for r in result:
        info.append([r.columns.tolist(), r.dtypes.values, [len(r)]])
    if len(result) > nfrag:
        raise Exception("Error in parallelize function")

    return result, info


