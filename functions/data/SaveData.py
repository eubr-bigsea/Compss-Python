#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__ = "Lucas Miguel S Ponce"
__email__  = "lucasmsp@gmail.com"

import time
import math
import numpy as np
import pandas as pd

from pycompss.api.task          import task
from pycompss.api.parameter     import *



#------------------------------------------------------------------------------
# Save Methods


# APPEND VAI FICAR NO MASTER

def SaveOperation(data,settings,numFrag):
	format_file = settings['format']
	filename = settings['filename']

	if format_file == 'CSV':
		mode = settings['mode']
		header = settings['header']
		for f in range(numFrag):
			output = "{}_part{}".format(filename,f)
			tmp = SaveToCSV(output, data[f], mode, header)

	elif format_file == 'JSON':
		for f in range(numFrag):
			output = "{}_part{}".format(filename,f)
			tmp = SaveToJson(output,data[f])




@task(filename = FILE_OUT)
def SaveToCSV(filename,data,mode,header):
    """
        SaveToCSV():

        Method used to save a dataframe into a file (CSV).

        :param filename: The name used in the output.
        :param data: The pandas dataframe which you want to save.
        :param mode: append, overwrite, ignore or error

    """
    import os.path


    if mode is 'append':
        mode = 'a'
    elif mode is 'ignore':
        if os.path.exists(filename):
            return None
    elif mode is 'error':
        if os.path.exists(filename):
            return None    # !   TO DO: RAISE SOME ERROR
    else:
        mode = 'w'

    print data
    if len(data)==0:
        data = pd.DataFrame()
    if header:
        data.to_csv(filename,sep=',',mode=mode, header=True,index=False)
    else:
        data.to_csv(filename,sep=',',mode=mode, header=False,index=False)

    return None


def SaveToPickle(outfile, data):
    """
        Save an array to a serizable Pickle file format

        :param outfile: the /path/file.npy
        :param data: the data to save
    """
    with open(outfile, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

@task(filename = FILE_OUT)
def SaveToJson(filename,data):
	"""
	    SaveToJson():

	    Method used to save a dataframe into a JSON (following the
		'records' pandas orientation).

	    :param filename: The name used in the output.
	    :param data: The pandas dataframe which you want to save.
	    :param mode: append, overwrite, ignore or error

	"""
	data.to_json(filename, orient='records')
	return None
