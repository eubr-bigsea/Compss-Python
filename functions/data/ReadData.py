# -*- coding: utf-8 -*-
#!/usr/bin/env python


from pycompss.api.task          import task
from pycompss.api.parameter     import *



def ReadCSVOperation(filename,settings):
    """
        ReadCSVOperation:

        Method used to load a pandas DataFrame from a csv file.

        :param filename:        The absolute path where the dataset is stored.
                                Each dataset is expected to be in a specific folder.
                                The folder will have the name of the dataset with the suffix "_folder".
                                The dataset is already divided into numFrags files, sorted lexicographically.
        :param settings:        A dictionary with the following parameters:
          - 'separator':        The string used to separate values (default, ',');
          - 'header':           True if the first line is a header, otherwise is False (default, True);
          - 'infer':
            * "NO":              Do not infer the data type of each column (will be string);
            * "FROM_VALUES":     Try to infer the data type of each column (default);
            * "FROM_LIMONERO":   !! NOT IMPLEMENTED YET!!
          - 'na_values':         A list with the all nan caracteres to be considerated.
                                 Default list:   '', '#N/A', '#N/A N/A', '#NA', '-1.#IND',
                                                 '-1.#QNAN', '-NaN', '-nan', '1.#IND', '1.#QNAN',
                                                 'N/A', 'NA', 'NULL', 'NaN', 'nan'
        :return                  A DataFrame splitted in a list with length N.

        Example:
        $ ls /var/workspace/dataset_folder
            dataset_00     dataset_02
            dataset_01     dataset_03
    """
    separator = settings.get('separator',',')
    header    = settings.get('header', True)
    infer     = settings.get('infer', 'FROM_VALUES')
    na_values = ['', '#N/A', '#N/A N/A', '#NA', '-1.#IND',
                 '-1.#QNAN', '-NaN', '-nan', '1.#IND', '1.#QNAN',
                 'N/A', 'NA', 'NULL', 'NaN', 'nan']
    if 'na_values' in settings:
        na_values = settings['na_values']
    if infer == "FROM_LIMONERO":
        infer = "FROM_VALUES"

    import os, os.path
    data = []
    DIR = filename+"_folder"

    #BUG: COMPSs dont handle with a string "\n" as a parameter, using the tag "<new_line>" instead.
    data =  [ ReadCSV( "{}/{}".format(DIR,name),
                        separator,
                        header,
                        infer,
                        na_values) for name in sorted(os.listdir(DIR))
            ]

    return data

@task(returns=list, filename = FILE_IN)
def ReadCSV(filename,separator,header,infer,na_values):
    import pandas as pd
    if separator == "<new_line>": separator = "\n"

    if infer =="NO":
        if header:
            df = pd.read_csv(filename,sep=separator,na_values=na_values,dtype='str');
        else:
            df = pd.read_csv(filename,sep=separator,na_values=na_values,header=0,dtype='str');

    elif infer == "FROM_VALUES":
        if header:
            df = pd.read_csv(filename,sep=separator,na_values=na_values);
        else:
            df = pd.read_csv(filename,sep=separator,na_values=na_values,header=0);

    elif infer == "FROM_LIMONERO":
        df = pd.DataFrame([])

    return df

#-------------------------------------------------------------

def ReadJsonOperation(filename, settings):
    """
        ReadCSVOperation():

        Method used to load a pandas DataFrame from a json file  (following the
		'records' pandas orientation).

        :param filename:        The absolute path where the dataset is stored.
                                Each dataset is expected to be in a specific folder.
                                The folder will have the name of the dataset with the suffix "_folder".
                                The dataset is already divided into numFrags files, sorted lexicographically.
        :param settings:        A dictionary with the following parameters:
          - 'infer':
            * "NO":              Do not infer the data type of each column (will be string);
            * "FROM_VALUES":     Try to infer the data type of each column (default);
            * "FROM_LIMONERO":   !! NOT IMPLEMENTED YET!!
        :return                  A DataFrame splitted in a list with length N.

        Example:
        $ ls /var/workspace/dataset_folder
            dataset_00     dataset_02
            dataset_01     dataset_03
    """
    infer     = settings.get('infer', 'FROM_VALUES')
    if infer == "FROM_LIMONERO":
        infer = "FROM_VALUES"

    import os, os.path
    data = []
    DIR = filename+"_folder"

    data =  [ReadJson(  "{}/{}".format(DIR,name),
                        infer) for name in sorted(os.listdir(DIR))]

    return data

#-------------------------------------------------------------

@task(returns=list, filename = FILE_IN)
def ReadJson(filename, infer):
    """
	    ReadJson():

	    Method used to read a file as a dataframe (following the
		'records' pandas orientation).
	    :param filename: The name used in the output.
	    :param infer:
            - "NO":             Do not infer the data type of each column (will be string);
            - "FROM_VALUES":    Try to infer the data type of each column;
            - "FROM_LIMONERO":  !! NOT IMPLEMENTED YET!!
        return:       The partial pandas dataframe.
    """
    import pandas as pd
    if infer == "NO":
        df = pd.read_json(filename, orient='records', dtype='str')
    else:
        df = pd.read_json(filename, orient='records')
	return df



def ReadCSVFromHDFSOperation(settings, numFrag):
    """
    Reads a tabular file (like csv) from HDFS and split it into numFrag parts.

    :param settings:        A dictionary with the following parameters:
      - path:               The path of the file from the / of the HDFS;
      - host:               The host of the Namenode HDFS; (default, localhost)
      - port:               Port of the Namenode HDFS; (default, 9000)

      - 'separator':        The string used to separate values (default, ',');
      - 'header':           True if the first line is a header, otherwise is False (default, True);
      - 'infer':
        * "NO":              Do not infer the data type of each column (will be string);
        * "FROM_VALUES":     Try to infer the data type of each column (default);
        * "FROM_LIMONERO":   !! NOT IMPLEMENTED YET!!
      - 'na_values':         A list with the all nan caracteres to be considerated.
                             Default list:   '', '#N/A', '#N/A N/A', '#NA', '-1.#IND',
                                             '-1.#QNAN', '-NaN', '-nan', '1.#IND', '1.#QNAN',
                                             'N/A', 'NA', 'NULL', 'NaN', 'nan'
    :param numFrag:   A number of fragments;
    :return           A DataFrame splitted in a list with length N.
    """
    import hdfs_pycompss.hdfsConnector as hdfs

    HDFS_BLOCKS = hdfs.findNBlocks(settings, numFrag)

    infer = settings.get('infer', 'FROM_VALUES')
    if infer == 'FROM_VALUES':
        settings['infer'] = True
    elif infer == 'FROM_LIMONERO':
        # TODO: integrate to Limonero
        settings['infer'] = False
    else:
        # 'NO':
        settings['infer'] = False

    data = [readHDFSData(block,settings) for block in HDFS_BLOCKS ]
    return data


@task(returns=list)
def readHDFSData(block, csv_options):
    import hdfs_pycompss.hdfsConnector as hdfs
    return hdfs.readDataFrame(block, csv_options)
