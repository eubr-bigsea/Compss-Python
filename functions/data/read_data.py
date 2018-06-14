#!/usr/bin/python
# -*- coding: utf-8 -*-
"""ReadData operations.

 - ReadOperationFs: Method used to load a pandas DataFrame in parallel from a
csv or json file (following the 'records' pandas orientation) splitted in
many other files, both from commom fs.

 - ReadOperationHDFS: Method used to load a pandas DataFrame in parallel from a
csv or json file (following the 'records' pandas orientation) splitted in
many other files, both from HDFS.

 - ReadOneCSVOperation: Method used to load a pandas DataFrame from a
unique csv file.

"""

from pycompss.api.task import task
from pycompss.api.parameter import FILE_IN
from hdfspycompss.Block import Block
from hdfspycompss.HDFS import HDFS
import pandas as pd


class ReadOperationFs(object):
    """Method to load a pandas DataFrame in parallel from a csv or json file.

        :param filename: The absolute path where the dataset is stored. Each
        dataset is expected to be in a specific folder. The folder will have
        the name of the dataset with the suffix "_folder".
        NOTE: It assumes that dataset is already divided into N files.
        :param settings: A dictionary with the following parameters:
          - 'format': File format, csv or json
          - 'infer':
            *"NO": Do not infer the data type of each field;
            *"FROM_VALUES": Try to infer the data type of each field (default);
            *"FROM_LIMONERO": !! NOT IMPLEMENTED YET!!
          - to CSV files:
            - 'separator': Value used to separate fields (default, ',');
            - 'header': True (default) if the first line is a header;
            - 'na_values': A list with the all nan caracteres. Default list:
                '', '#N/A', '#N/A N/A', '#NA', '-1.#IND', '-1.#QNAN', '-NaN',
                '-nan', '1.#IND', '1.#QNAN', 'N/A', 'NA', 'NULL', 'NaN', 'nan'
        :return A DataFrame splitted in a list with length N.

        Example:
        $ ls /var/workspace/dataset_folder
            dataset_00     dataset_02
            dataset_01     dataset_03
        """

    def transform(self, filename, settings):

        settings = self.preprocessing(settings)
        import os.path
        path = filename + "_folder"
        files = sorted(os.listdir(path))
        result = [[] for _ in range(len(files))]
        for f, name in enumerate(files):
            obj = "{}/{}".format(path, name)
            result[f] = _read_from_fs(obj, settings)

        return result

    def preprocessing(self, settings):
        format_type = settings.get('format', 'csv')
        if format_type is 'csv':
            settings['separator'] = settings.get('separator', ',')
            settings['header'] = settings.get('header', True)
            settings['infer'] = settings.get('infer', 'FROM_VALUES')
            na_values = ['', '#N/A', '#N/A N/A', '#NA', '-1.#IND',
                         '-1.#QNAN', '-NaN', '-nan', '1.#IND', '1.#QNAN',
                         'N/A', 'NA', 'NULL', 'NaN', 'nan']
            if 'na_values' not in settings:
                settings['na_values'] = na_values
            if settings['infer'] == "FROM_LIMONERO":
                settings['infer'] = "FROM_VALUES"

            if settings['infer'] not in ['NO', 'FROM_VALUES', 'FROM_LIMONERO']:
                raise Exception("Please inform a valid `infer` type.")

            # BUG: There is a COMPSs bug when separator is "\n".
            # Because of that,  use a mask "<new_line>" instead in these cases.
            if settings['separator'] == '\n':
                settings['separator'] = '<new_line>'

        elif format_type is 'json':
            settings['infer'] = settings.get('infer', 'FROM_VALUES')
            if settings['infer'] == "FROM_LIMONERO":
                settings['infer'] = "FROM_VALUES"
        else:
            raise Exception("Please inform a valid `format` type.")

        return settings


@task(returns=list, filename=FILE_IN)
def _read_from_fs(filename, settings):
    """Load a fragment of a csv or json file in a pandas DataFrame."""

    format_type = settings.get('format', 'csv')
    infer = settings['infer']

    if format_type is 'csv':
        separator = settings['separator']
        header = settings['header']
        na_values = settings['na_values']

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
    else:

        if infer == "NO":
            df = pd.read_json(filename, orient='records', dtype='str',
                              lines=True)
        else:
            df = pd.read_json(filename, orient='records', lines=True)

    return df


class ReadOperationHDFS(object):
    """Method to load a pandas DataFrame in parallel from a csv or json file.

        :param filename: The absolute path where the dataset is stored. Each
        dataset is expected to be in a specific folder. The folder will have
        the name of the dataset with the suffix "_folder".
        NOTE: It assumes that dataset is already divided into N files.
        :param settings: A dictionary with the following parameters:
          - 'format': File format, CSV or JSON
          - 'infer':
            *"NO": Do not infer the data type of each field;
            *"FROM_VALUES": Try to infer the data type of each field (default);
            *"FROM_LIMONERO": !! NOT IMPLEMENTED YET!!
          - to CSV files:
            - 'separator': Value used to separate fields (default, ',');
            - 'header': True (default) if the first line is a header;
            - 'na_values': A list with the all nan caracteres. Default list:
                '', '#N/A', '#N/A N/A', '#NA', '-1.#IND', '-1.#QNAN', '-NaN',
                '-nan', '1.#IND', '1.#QNAN', 'N/A', 'NA', 'NULL', 'NaN', 'nan'
        :return A DataFrame splitted in a list with length N.
        """

    def transform(self, filename, settings, nfrag):

        blocks, settings = self.preprocessing(filename, settings, nfrag)
        data = [[] for _ in blocks]
        for f, block in enumerate(blocks):
            data[f] = _read_from_hdfs(block, settings)

        return data

    def preprocessing(self, filename, settings, nfrag):
        format_type = settings.get('format', 'csv')

        if format_type is 'csv':
            settings['separator'] = settings.get('separator', ',')
            settings['header'] = settings.get('header', True)
            settings['infer'] = settings.get('infer', 'FROM_VALUES')
            na_values = ['', '#N/A', '#N/A N/A', '#NA', '-1.#IND',
                         '-1.#QNAN', '-NaN', '-nan', '1.#IND', '1.#QNAN',
                         'N/A', 'NA', 'NULL', 'NaN', 'nan']
            if 'na_values' not in settings:
                settings['na_values'] = na_values
            if settings['infer'] == "FROM_LIMONERO":
                settings['infer'] = "FROM_VALUES"

            if settings['infer'] not in ['NO', 'FROM_VALUES', 'FROM_LIMONERO']:
                raise Exception("Please inform a valid `infer` type.")

            # BUG: There is a COMPSs bug when separator is "\n".
            # Because of that,  use a mask "<new_line>" instead in these cases.
            if settings['separator'] == '\n':
                settings['separator'] = '<new_line>'

        elif format_type is 'json':
            settings['infer'] = settings.get('infer', 'FROM_VALUES')
            if settings['infer'] == "FROM_LIMONERO":
                settings['infer'] = "FROM_VALUES"
        else:
            raise Exception("Please inform a valid `format` type.")

        host = settings['host']
        port = settings['port']
        if nfrag > 0:
            blocks = HDFS(host=host, port=port).findNBlocks(filename, nfrag)
        else:
            blocks = HDFS(host=host, port=port).findBlocks(filename)

        return blocks, settings

    def transform_serial(self, blk, csv_options):
        return _read_from_hdfs_(blk, csv_options)


@task(returns=list)
def _read_from_hdfs(blk, csv_options):
    """Load a dataframe from a HDFS file."""
    print "[INFO - ReadOperationHDFS] - ", blk
    if csv_options.get('separator', '') == '<new_line>':
        csv_options['separator'] = '\n'
    return _read_from_hdfs_(blk, csv_options)


def _read_from_hdfs_(blk, csv_options):
    """Load a dataframe from a HDFS file."""
    df = Block(blk).readDataFrame(csv_options)
    # df = df.infer_objects()
    print "[INFO - ReadOperationHDFS] - ", df.info(verbose=False)
    return df


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
