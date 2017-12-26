#!/usr/bin/python
# -*- coding: utf-8 -*-
"""ReadData operations.

 - ReadCSVOperation: Method used to load a pandas DataFrame in parallel from a
csv file splitted in many other files.

 - ReadJsonOperation: Method used to load a pandas DataFrame in parallel from
a json file (following the 'records' pandas orientation) splitted in many
other files.

 - ReadCSVFromHDFSOperation: Method used to load a pandas DataFrame in
parallel from a csv file in HDFS.

 - ReadJsonFromHDFSOperation: Method used to load a pandas DataFrame in
parallel from a json file (following the 'records' pandas orientation) in HDFS.

 - ReadOneCSVOperation: Method used to load a pandas DataFrame from a
unique csv file.

"""

from pycompss.api.task import task
from pycompss.api.parameter import *


def ReadCSVOperation(filename, settings):
    """Method used to load a pandas DataFrame in parallel from a csv file.

    :param filename: The absolute path where the dataset is stored. Each
    dataset is expected to be in a specific folder. The folder will have
    the name of the dataset with the suffix "_folder". The dataset is already
    divided into numFrags files.
    :param settings: A dictionary with the following parameters:
      - 'separator': Value used to separate fields (default, ',');
      - 'header': True (default) if the first line is a header;
      - 'infer':
        *"NO": Do not infer the data type of each field;
        *"FROM_VALUES": Try to infer the data type of each field (default);
        *"FROM_LIMONERO": !! NOT IMPLEMENTED YET!!
      - 'na_values': A list with the all nan caracteres. Default list:
        '', '#N/A', '#N/A N/A', '#NA', '-1.#IND', '-1.#QNAN', '-NaN',
        '-nan', '1.#IND', '1.#QNAN', 'N/A', 'NA', 'NULL', 'NaN', 'nan'
    :return A DataFrame splitted in a list with length N.

    Example:
    $ ls /var/workspace/dataset_folder
        dataset_00     dataset_02
        dataset_01     dataset_03
    """
    separator = settings.get('separator', ',')
    header = settings.get('header', True)
    infer = settings.get('infer', 'FROM_VALUES')
    na_values = ['', '#N/A', '#N/A N/A', '#NA', '-1.#IND',
                 '-1.#QNAN', '-NaN', '-nan', '1.#IND', '1.#QNAN',
                 'N/A', 'NA', 'NULL', 'NaN', 'nan']
    if 'na_values' in settings:
        na_values = settings['na_values']
    if infer == "FROM_LIMONERO":
        infer = "FROM_VALUES"

    import os
    import os.path
    DIR = filename+"_folder"

    if infer not in ['NO', 'FROM_VALUES', 'FROM_LIMONERO']:
        raise Exception("Please inform a valid `infer` type.")

    # BUG: There is a COMPSs bug whent separator is "\n". Because of that,
    #     use a mask "<new_line>" instead in these cases.
    if separator == '\n':
        separator = '<new_line>'

    files = sorted(os.listdir(DIR))
    result = [[] for f in range(len(files))]
    for f, name in enumerate(files):
        result[f] = ReadCSV("{}/{}".format(DIR, name),
                            separator,
                            header,
                            infer,
                            na_values)
    return result


@task(returns=list, filename=FILE_IN)
def ReadCSV(filename, separator, header, infer, na_values):
    """Load a fragment of a csv file in a pandas DataFrame."""
    import pandas as pd
    if separator == "<new_line>":
        separator = "\n"
    if header:
        header = 'infer'
    else:
        header = None

    if infer == "NO":
        df = pd.read_csv(filename, sep=separator, na_values=na_values,
                         header=header, dtype='str')

    elif infer == "FROM_VALUES":
        df = pd.read_csv(filename, sep=separator, na_values=na_values,
                         header=header)

    elif infer == "FROM_LIMONERO":
        df = pd.DataFrame([])

    if not header:
        n_cols = len(df.columns)
        new_columns = ['col_{}'.format(i) for i in range(n_cols)]
        df.columns = new_columns

    return df

# -----------------------------------------------------------------------------

def ReadJsonOperation(filename, settings):
    """Method used to load a pandas DataFrame in parallel from a json file.

    :param filename: The absolute path where the dataset is stored. Each
    dataset is expected to be in a specific folder. The folder will have
    the name of the dataset with the suffix "_folder". The dataset is already
    divided into numFrags files.
    :param settings: A dictionary with the following parameters:
      - 'infer':
        *"NO": Do not infer the data type of each column;
        *"FROM_VALUES": Try to infer the data type of each field (default);
        *"FROM_LIMONERO": !! NOT IMPLEMENTED YET!!
    :return A DataFrame splitted in a list with length N.

    Example:
    $ ls /var/workspace/dataset_folder
        dataset_00     dataset_02
        dataset_01     dataset_03
    """
    infer = settings.get('infer', 'FROM_VALUES')
    if infer == "FROM_LIMONERO":
        infer = "FROM_VALUES"

    import os
    import os.path

    DIR = filename+"_folder"
    files = sorted(os.listdir(DIR))
    result = [[] for f in range(len(files))]
    for f, name in enumerate(files):
        result[f] = ReadJson("{}/{}".format(DIR, name), infer)

    return result


@task(returns=list, filename=FILE_IN)
def ReadJson(filename, infer):
    """Load a json ('records' pandas orientation) as a dataframe."""
    import pandas as pd
    if infer == "NO":
        df = pd.read_json(filename, orient='records', dtype='str', lines=True)
    else:
        df = pd.read_json(filename, orient='records', lines=True)
    return df

# -----------------------------------------------------------------------------


def ReadCSVFromHDFSOperation(settings, numFrag):
    """Method used to load a pandas DataFrame in parallel from a csv in HDFS.

    :param settings: A dictionary with the following parameters:
      - path: The path of the file from the / of the HDFS;
      - host: The host of the Namenode HDFS; (default, localhost)
      - port: Port of the Namenode HDFS; (default, 9000)
      - 'separator': The string used to separate values (default, ',');
      - 'header': True (default) if the first line is a header;
      - 'infer':
        * "NO": Do not infer the data type of each column;
        * "FROM_VALUES": Try to infer the data type of each column (default);
        * "FROM_LIMONERO": !! NOT IMPLEMENTED YET!!
      - 'na_values': A list with the all nan caracteres. Default list:
        '', '#N/A', '#N/A N/A', '#NA', '-1.#IND', '-1.#QNAN', '-NaN',
        '-nan', '1.#IND', '1.#QNAN', 'N/A', 'NA', 'NULL', 'NaN', 'nan'
      - 'mode':
        * 'FAILFAST': Stop processing and raise error;
        * 'DROPMALFORMED': Ignore whole corrupted record;
    :param numFrag: A number of fragments;
    :return A DataFrame splitted in a list with length N.
    """
    import hdfs_pycompss.hdfsConnector as hdfs

    HDFS_BLOCKS = hdfs.findNBlocks(settings, numFrag)

    infer = settings.get('infer', 'FROM_VALUES')
    if infer not in ['NO', 'FROM_VALUES', 'FROM_LIMONERO']:
        raise Exception("Please inform a valid `infer` type.")

    if infer == 'FROM_VALUES':
        settings['infer'] = True
    elif infer == 'FROM_LIMONERO':
        # TODO: integrate to Limonero
        settings['infer'] = False
    else:
        # 'NO':
        settings['infer'] = False

    data = [readHDFSData(block, settings) for block in HDFS_BLOCKS]
    return data


@task(returns=list)
def readHDFSData(block, csv_options):
    """Receive part of a dataframe in the HDFS."""
    import hdfs_pycompss.hdfsConnector as hdfs
    df = hdfs.readDataFrame(block, csv_options)
    # df = df.infer_objects()
    return df

# -----------------------------------------------------------------------


def ReadJsonFromHDFSOperation(settings, numFrag):
    """Read a json file in parallel from HDFS to a pandas DataFrame list.

    :param settings: A dictionary with the following parameters:
      - path: The path of the file from the / of the HDFS;
      - host: The host of the Namenode HDFS; (default, localhost)
      - port: Port of the Namenode HDFS; (default, 9000)

      - 'separator': The string used to separate values (default, ',');
      - 'header': True (default) if the first line is a header;
      - 'infer':
        * "NO": Do not infer the data type of each column;
        * "FROM_VALUES":  Try to infer the data type of each column (default);
        * "FROM_LIMONERO": !! NOT IMPLEMENTED YET!!
      - 'na_values': A list with the all nan caracteres. Default list:
        '', '#N/A', '#N/A N/A', '#NA', '-1.#IND', '-1.#QNAN', '-NaN',
        '-nan', '1.#IND', '1.#QNAN', 'N/A', 'NA', 'NULL', 'NaN', 'nan'
      - 'mode':
        * 'FAILFAST': Stop processing and raise error
        * 'DROPMALFORMED': Ignore whole corrupted record
    :param numFrag: A number of fragments;
    :return A DataFrame splitted in a list with length N.
    """
    import hdfs_pycompss.hdfsConnector as hdfs

    HDFS_BLOCKS = hdfs.findNBlocks(settings, numFrag)

    infer = settings.get('infer', 'FROM_VALUES')
    if infer not in ['NO', 'FROM_VALUES', 'FROM_LIMONERO']:
        raise Exception("Please inform a valid `infer` type.")

    if infer == 'FROM_VALUES':
        settings['infer'] = True
    elif infer == 'FROM_LIMONERO':
        # TODO: integrate to Limonero
        settings['infer'] = False
    else:
        # 'NO':
        settings['infer'] = False

    data = [readHDFSJson(block, settings) for block in HDFS_BLOCKS]
    return data


@task(returns=list)
def readHDFSJson(block, csv_options):
    """Load a fragment of json in a HDFS to a pandas DataFrame."""
    import hdfs_pycompss.hdfsConnector as hdfs
    return hdfs.readJsonDataFrame(block, csv_options)

# ----------------------------------------------------------------------


def ReadOneCSVOperation(filename, settings):
    """Method used to load a pandas DataFrame from a unique csv file.

    :param filename:  The absolute path where the dataset is stored.
    :param settings: A dictionary with the following parameters:
      - 'separator': Value used to separate fields (default, ',');
      - 'header': True (default) if the first line is a header;
      - 'infer':
        *"NO": Do not infer the data type of each field;
        *"FROM_VALUES": Try to infer the data type of each field (default);
        *"FROM_LIMONERO": !! NOT IMPLEMENTED YET!!
      - 'na_values': A list with the all nan caracteres. Default list:
        '', '#N/A', '#N/A N/A', '#NA', '-1.#IND', '-1.#QNAN', '-NaN', '-nan',
        '1.#IND', '1.#QNAN', 'N/A', 'NA', 'NULL', 'NaN', 'nan'
    :return A DataFrame splitted in a list with length N.
    """
    separator = settings.get('separator', ',')
    header = settings.get('header', True)
    infer = settings.get('infer', 'FROM_VALUES')
    na_values = ['', '#N/A', '#N/A N/A', '#NA', '-1.#IND',
                 '-1.#QNAN', '-NaN', '-nan', '1.#IND', '1.#QNAN',
                 'N/A', 'NA', 'NULL', 'NaN', 'nan']
    if 'na_values' in settings:
        na_values = settings['na_values']
    if infer == "FROM_LIMONERO":
        infer = "FROM_VALUES"

    if infer not in ['NO', 'FROM_VALUES', 'FROM_LIMONERO']:
        raise Exception("Please inform a valid `infer` type.")

    import pandas as pd
    if separator == "<new_line>":
        separator = "\n"

    if header:
        header = 'infer'
    else:
        header = None

    if infer == "NO":
        df = pd.read_csv(filename, sep=separator, na_values=na_values,
                         header=header, dtype='str')

    elif infer == "FROM_VALUES":
        df = pd.read_csv(filename, sep=separator, na_values=na_values,
                         header=header)

    elif infer == "FROM_LIMONERO":
        df = pd.DataFrame([])

    if not header:
        n_cols = len(df.columns)
        new_columns = ['col_{}'.format(i) for i in range(n_cols)]
        df.columns = new_columns
    return df
