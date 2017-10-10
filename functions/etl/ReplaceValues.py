# -*- coding: utf-8 -*-
#!/usr/bin/env python

__author__ = "Lucas Miguel S Ponce"
__email__  = "lucasmsp@gmail.com"

from pycompss.api.task          import task
from pycompss.api.parameter     import *

import numpy  as np
import pandas as pd


def  ReplaceValuesOperation (data,settings,numFrag):
	"""
	ReplaceValuesOperation():

	Replace one or more values to new ones in a pandas's dataframe.

     :param data:      	A list with numFrag pandas's dataframe;
     :param settings:   A dictionary that contains:
		- replaces:	    A dictionary where each key is a column to perform
						an operation. Each key is linked to a matrix of 2xN.
						The first row is respect to the old values (or a regex)
						and the last is the new values.
		- regex:		True, to use a regex expression, otherwise is False.
		                Can be used only if the columns are strings types.
						(default is False);
	 :param numFrag:    The number of fragments;
	 :return:           Returns a list with numFrag pandas's dataframe

	example:
		* settings['replaces'] = {
		'Col1':[[<old_value1>,<old_value2>],[<new_value1>,<new_value2>]],
		'Col2':[[<old_value3>],[<new_value3>]]
		}

	"""

	Validate(settings)

	for f in range(numFrag):
		data[f] = ReplaceValues_p(data[f], settings)
	return data


def Validate(settings):
	replaces = settings.get('replaces',{})
	WRONG = False
	for col in replaces:
		if len(replaces[col]) != 2:
			WRONG = True
			break
		if len(replaces[col][0]) != len(replaces[col][1]):
			WRONG = True

	if len(replaces) == 0 or WRONG:
		raise Exception('You must inform a valid replaces settings !')

@task(returns=list)
def ReplaceValues_p(data, settings):
    dict_replaces = settings['replaces']
    regexes = settings.get('regex', False) # only if is string

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


	        # replace might not work with floats because the floating point
	        # representation you see in the repr of the DataFrame might not be
	        # the same as the underlying float. Because of that, we need to
			# perform float operations in separate way.

            for old,new in zip(tmp_o,tmp_v):
                mask = np.isclose(data[col],  old,rtol=1e-06)
                data.ix[mask, col] = new



        data[col].replace(	to_replace=olds_v, value=news_v,
							regex=regexes, inplace=True
							)

    return data
