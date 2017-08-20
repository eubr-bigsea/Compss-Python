# -*- coding: utf-8 -*-
#!/usr/bin/env python


from pycompss.functions.data import chunks
from pycompss.api.task          import task
from pycompss.api.parameter     import *
from pycompss.functions.reduce import mergeReduce

import numpy as np
import math
import pickle
import pandas as pd


def  ReplaceValuesOperation (data,settings,numFrag):
    for f in range(numFrag):
        data[f] = ReplaceValues_p(data[f], settings)
    return data


@task(returns=list)
def ReplaceValues_p(data, settings):
    dict_replaces = settings['replaces']
    regexes = settings['regex'] # only if is string

    for col in dict_replaces:
        olds_v, news_v = dict_replaces[col]
        if not regexes:
            tmp_o = []
            tmp_v = []
            ixs = []
            for ix in range(len(olds_v)):
                if isinstance(olds_v[ix], float):
                    tmp_o.append(olds_v[ix])
                    tmp_v.append(news_v[ix])
                    ixs.append(ix)
            olds_v = [olds_v[ix] for ix in range(len(olds_v)) if ix not in ixs]
            news_v = [news_v[ix] for ix in range(len(news_v)) if ix not in ixs]

            for old,new in zip(tmp_o,tmp_v):
                #df.loc[<row selection>, <column selection>]
                mask = np.isclose(data[col],  old)
                data.ix[mask, col] = new

        # replace might not work with floats because the floating point
        # representation you see in the repr of the DataFrame might not be
        # the same as the underlying float. Because of that, we need to perform
        # float operations in separate way.

        data[col].replace(to_replace=olds_v, value=news_v,regex=regexes,inplace=True)

    return data
