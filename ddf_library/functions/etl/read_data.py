#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from ddf_library.utils import generate_info, merge_info
from .parallelize import parallelize

from pycompss.api.task import task

import pandas as pd


class DataReader(object):
    """
    Loads data from a csv, txt or json file and saves it
    in a list of n fragments of pandas DataFrame.
    """

    def __init__(self, filepath, nfrag=None, format='csv', storage='hdfs',
                 distributed=True, dtype=None, separator=',',
                 error_bad_lines=True, header=True, na_values=None,
                 host='localhost', port=9000):
        """
        :param filepath: The absolute path where the data set is stored;
        :param nfrag: Number of partitions to split the loaded data,
         if distributed option is False (default, number of cpu);
        :param format: File format, csv, json or txt;
        :param storage: Where the file is, `fs` to common file system or
         `hdfs`. Default is `hdfs`;
        :param distributed: if the absolute path represents a unique file or
         a folder with multiple files;
        :param dtype: Type name or dict of column (default, 'str'). Data type
         for data or columns. E.g. {‘a’: np.float64, ‘b’: np.int32, ‘c’:
         ‘Int64’} Use str or object together with suitable na_values settings
         to preserve and not interpret dtype;
        :param separator: Value used to separate fields (default, ',');
        :param error_bad_lines: Lines with too many fields (e.g. a csv line
         with too many commas) will by default cause an exception to be raised,
         and no DataFrame will be returned. If False, then these “bad lines”
         will dropped from the DataFrame that is returned.
        :param header:  True (default) if the first line is a header;
        :param na_values: A list with the all nan characters. Default list:
         ['', '#N/A', '#N/A N/A', '#NA', '-1.#IND', '-1.#QNAN', '-NaN',
         '-nan', '1.#IND', '1.#QNAN', 'N/A', 'NA', 'NULL', 'NaN', 'nan']
        :param host: Namenode host if storage is `hdfs` (default, 'localhost');
        :param port: Port to Namenode host if storage is `hdfs` (default, 9000);

        :return A list with length N.


        ..see also: `Dtype information <https://docs.scipy.org/doc/numpy-1.15
         .0/reference/arrays.dtypes.html>`_.
        """

        if format not in ['csv', 'json', 'txt']:
            raise Exception("Only `csv`, `json` and `txt` are supported.")

        if storage not in ['fs', 'hdfs']:
            raise Exception("Only `fs` and `hdfs` are supported.")

        if nfrag is None:
            import multiprocessing
            nfrag = multiprocessing.cpu_count()

        if dtype is None:
            dtype = 'str'

        if na_values is None:
            na_values = ['', '#N/A', '#N/A N/A', '#NA', '-1.#IND',
                         '-1.#QNAN', '-NaN', '-nan', '1.#IND', '1.#QNAN',
                         'N/A', 'NA', 'NULL', 'NaN', 'nan']

        self.filepath = filepath
        self.nfrag = nfrag
        self.format = format
        self.storage = storage
        self.distributed = distributed
        self.dtype = [dtype]
        self.na_values = na_values
        self.error_bad_lines = error_bad_lines
        self.host = host
        self.port = port

        if self.format == 'txt':
            separator = '\n'
            header = None

        # if format file are `csv` or `txt`
        self.separator = [separator]
        self.header = header if header else None

        if self.storage == 'hdfs':
            self.blocks = self.preprocessing_hdfs()
        else:
            self.blocks = self.preprocessing_fs()

    def preprocessing_hdfs(self):
        from hdfspycompss.hdfs import HDFS

        if not self.distributed:

            if self.nfrag > 0:
                blocks = HDFS(host=self.host, port=self.port)\
                    .find_n_blocks(self.filepath, self.nfrag)
            else:
                blocks = HDFS(host=self.host, port=self.port)\
                    .find_blocks(self.filepath)
        else:
            # Get a list of blocks as files
            hdfs_client = HDFS(host=self.host, port=self.port)
            files_list = hdfs_client.ls(self.filepath)
            blocks = [hdfs_client.find_n_blocks(f, 1)[0] for f in files_list]

        return blocks

    def preprocessing_fs(self):

        blocks = []
        if self.distributed:
            import os.path
            files = sorted(os.listdir(self.filepath))
            for f, name in enumerate(files):
                obj = "{}/{}".format(self.filepath, name)
                blocks.append(obj)
        else:
            blocks.append(self.filepath)

        return blocks

    def get_blocks(self):
        return self.blocks

    def transform_fs_single(self):

        block = self.blocks[0]

        result, _ = _read_fs(block, self.format,
                             self.separator,
                             self.header, self.na_values,
                             self.dtype, self.error_bad_lines, 0)
        result, info = parallelize(result, self.nfrag)

        return result, info

    def transform_fs_distributed(self, block, params):

        frag = params['id_frag']

        result, info = _read_fs(block, self.format, self.separator,
                                self.header, self.na_values, self.dtype,
                                self.error_bad_lines, frag)

        return result, info

    def transform_hdfs(self, block, params):

        frag = params['id_frag']

        result, info = _read_hdfs(block, self.format, self.separator,
                                  self.header, self.na_values,
                                  self.dtype, self.error_bad_lines, frag)

        return result, info


def _read_fs(filename, format_type, separator, header, na_values,
             dtype, error_bad_lines, frag):
    """Load a fragment of a csv or json file in a pandas DataFrame."""

    if format_type in ['csv', 'txt']:
        separator = separator[0]
        if header:
            header = 'infer'

        df = pd.read_csv(filename, sep=separator, na_values=na_values,
                         header=header, dtype=dtype[0],
                         error_bad_lines=error_bad_lines)

        if not header:
            n_cols = len(df.columns)
            new_columns = ['col_{}'.format(i) for i in range(n_cols)]
            df.columns = new_columns

    else:

        df = pd.read_json(filename, orient='records', dtype=dtype[0],
                          lines=True)

    info = generate_info(df, frag)
    return df, info


def _read_hdfs(blk, format_type, separator, header, na_values, dtype,
               error_bad_lines, frag):
    """Load a DataFrame from a HDFS file."""
    print("[INFO - ReadOperationHDFS] - ", blk)
    from hdfspycompss.block import Block
    separator = separator[0]

    df = Block(blk).read_dataframe(format_file=format_type, infer=False,
                                   separator=separator, dtype=dtype[0],
                                   header=header, na_values=na_values,
                                   error_bad_lines=error_bad_lines)

    info = generate_info(df, frag)
    return df, info
