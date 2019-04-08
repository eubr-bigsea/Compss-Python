#!/usr/bin/env python3
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

from ddf_library.utils import generate_info, merge_info
from .parallelize import parallelize
from pycompss.api.task import task
from pycompss.api.parameter import FILE_IN
import pandas as pd


class DataReader(object):

    def __init__(self, filepath, nfrag=4, format='csv', storage='hdfs',
                 distributed=True, dtype='str', separator=',',
                 error_bad_lines=True, header=True, na_values='',
                 host='localhost', port=9000):
        """
        Loads a pandas DataFrame from a csv, txt or json file.

        :param filepath: The absolute path where the dataset is stored;
        :param nfrag: Number of partitions to split the loaded data,
         if distribuited option is False (default, 4);
        :param format: File format, csv, json or txt;
        :param storage: Where the file is, `fs` to commom file system or
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
        :param na_values: A list with the all nan caracteres. Default list:
         ['', '#N/A', '#N/A N/A', '#NA', '-1.#IND', '-1.#QNAN', '-NaN',
         '-nan', '1.#IND', '1.#QNAN', 'N/A', 'NA', 'NULL', 'NaN', 'nan']
        :param host: Namenode host if storage is `hdfs` (default, 'localhost');
        :param port: Port to Namenode host if storage is `hdfs` (default, 9000);

        :return A DataFrame splitted in a list with length N.


        ..see also: `Dtype information <https://docs.scipy.org/doc/numpy-1.15
         .0/reference/arrays.dtypes.html>`_.

        """

        self.filepath = filepath
        self.nfrag = nfrag

        self.format = format
        if self.format not in ['csv', 'json', 'txt']:
            raise Exception("Only `csv`, `json` and `txt` are suppported.")

        self.storage = storage
        if self.storage not in ['fs', 'hdfs']:
            raise Exception("Only `fs` and `hdfs` are suppported.")

        self.distributed = distributed
        self.dtype = [dtype]

        # if format is `txt`
        if self.format is 'txt':
            separator = '\n'
            header = None

        # if format file are `csv` or `txt`
        self.separator = [separator]
        self.header = header if header else None

        na_values_default = ['', '#N/A', '#N/A N/A', '#NA', '-1.#IND',
                             '-1.#QNAN', '-NaN', '-nan', '1.#IND', '1.#QNAN',
                             'N/A', 'NA', 'NULL', 'NaN', 'nan']

        self.na_values = na_values if len(na_values) > 0 else na_values_default

        self.error_bad_lines = error_bad_lines
        # if storage is `hdfs`
        self.host = host
        self.port = port

        if self.storage is 'hdfs':
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
            # TODO
            blocks = []

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

    def transform(self, block=None):

        if block:
            blocks = [block]
        else:
            blocks = self.blocks

        result = [[] for _ in blocks]
        info = [[] for _ in blocks]

        if self.storage is 'fs':
            if self.distributed:

                for f, blk in enumerate(blocks):
                    result[f], info[f] = \
                        _read_fs_task(blk, self.format, self.separator,
                                      self.header, self.na_values, self.dtype,
                                      self.error_bad_lines, f)

                info = merge_info(info)
            else:
                result, info[0] = _read_fs(self.blocks[0], self.format,
                                           self.separator,
                                           self.header, self.na_values,
                                           self.dtype, self.error_bad_lines, 0)
                result, info = parallelize(result, self.nfrag)
        else:
            if not self.distributed:

                for f, block in enumerate(blocks):
                    result[f], info[f] = \
                        _read_hdfs_task(block, self.format, self.separator,
                                        self.header, self.na_values, self.dtype,
                                        self.error_bad_lines, f)

                info = merge_info(info)

            else:
                raise Exception("Not implemeted yet! ")

        return result, info

    def read_hdfs_serial(self, block, _):
        if not self.distributed:
            result, info = _read_hdfs(block, self.format, self.separator,
                                      self.header, self.na_values,
                                      self.dtype, self.error_bad_lines, 0)

        else:
            # TODO
            raise Exception("Not implemeted yet! ")

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
    """Load a dataframe from a HDFS file."""
    print("[INFO - ReadOperationHDFS] - ", blk)
    from hdfspycompss.block import Block
    separator = separator[0]

    df = Block(blk).read_dataframe(format_file=format_type, infer=False,
                                   separator=separator, dtype=dtype[0],
                                   header=header, na_values=na_values,
                                   error_bad_lines=error_bad_lines)

    info = generate_info(df, frag)
    return df, info


@task(returns=2, filename=FILE_IN)
def _read_fs_task(filename, format_type, separator, header,
                  na_values, dtype, error_bad_lines, frag):
    return _read_fs(filename, format_type, separator, header,
                    na_values, dtype, error_bad_lines, frag)


@task(returns=2)
def _read_hdfs_task(blk, format_type, separator, header,
                    na_values, dtype, error_bad_lines, frag):
    return _read_hdfs(blk, format_type, separator, header,
                      na_values, dtype, error_bad_lines, frag)
