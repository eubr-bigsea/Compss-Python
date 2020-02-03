#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from ddf_library.utils import generate_info
from .parallelize import parallelize

import pandas as pd


class DataReader(object):
    """
    Loads data from a csv, txt or json file and saves it
    in a list of n fragments of pandas DataFrame.
    """

    def __init__(self):
        self.filepath = None
        self.nfrag = None
        self.storage = None
        self.dtype = None
        self.host = None
        self.port = None
        self.distributed = None
        self.kwargs = None
        self.distributed = None
        self.blocks = None
        self.format = None
        self.header = None

    #TODO: infer
    def csv(self, filepath, num_of_parts='*', storage='hdfs', host='default',
            port=0, schema=None, sep=',', delimiter=None, na_filter=True,
            usecols=None, prefix=None, engine=None, converters=None,
            true_values=None, false_values=None, skipinitialspace=False,
            na_values=None, keep_default_na=True, skip_blank_lines=True,
            parse_dates=False, decimal='.', dayfirst=False, header=True,
            thousands=None, quotechar='"', doublequote=True, escapechar=None,
            comment=None, encoding=None,  error_bad_lines=True,
            warn_bad_lines=True, delim_whitespace=False, float_precision=None):

        if num_of_parts == '*':
            import multiprocessing
            num_of_parts = multiprocessing.cpu_count()

        if schema is None:
            schema = 'str'

        self.filepath = filepath
        self.nfrag = num_of_parts
        self.format = 'csv'
        self.storage = storage
        self.dtype = [schema]
        self.host = host
        self.port = port
        self.distributed = self.check_file_or_folder(filepath)
        self.header = header

        kwargs = dict()
        kwargs['float_precision'] = float_precision
        kwargs['dayfirst'] = dayfirst
        kwargs['encoding'] = encoding
        kwargs['converters'] = converters
        kwargs['escapechar'] = escapechar
        kwargs['parse_dates'] = parse_dates
        kwargs['na_values'] = na_values
        kwargs['quotechar'] = quotechar
        kwargs['doublequote'] = doublequote
        kwargs['error_bad_lines'] = error_bad_lines
        kwargs['warn_bad_lines'] = warn_bad_lines
        kwargs['sep'] = sep
        kwargs['prefix'] = prefix  # TODO
        kwargs['comment'] = comment
        kwargs['thousands'] = thousands
        kwargs['delim_whitespace'] = delim_whitespace
        kwargs['na_filter'] = na_filter
        kwargs['decimal'] = decimal
        kwargs['true_values'] = true_values
        kwargs['false_values'] = false_values
        kwargs['skipinitialspace'] = skipinitialspace
        kwargs['engine'] = engine
        kwargs['delimiter'] = delimiter
        kwargs['usecols'] = usecols
        kwargs['keep_default_na'] = keep_default_na
        kwargs['error_bad_lines'] = error_bad_lines
        kwargs['skip_blank_lines'] = skip_blank_lines
        self.kwargs = kwargs

        if self.storage == 'hdfs':
            self.blocks = self.preprocessing_hdfs()
        else:
            self.blocks = self.preprocessing_fs()

        return self

    def json(self, filepath, num_of_parts='*', storage='hdfs', host='default',
             port=0, schema=None, precise_float=False, encoding=None):

        if num_of_parts == '*':
            import multiprocessing
            num_of_parts = multiprocessing.cpu_count()

        if schema is None:
            schema = 'str'

        self.filepath = filepath
        self.nfrag = num_of_parts
        self.storage = storage
        self.dtype = [schema]
        self.host = host
        self.format = 'json'
        self.port = port
        self.distributed = self.check_file_or_folder(filepath)

        kwargs = dict()
        kwargs['precise_float'] = precise_float
        kwargs['encoding'] = encoding
        self.kwargs = kwargs

        if self.storage == 'hdfs':
            self.blocks = self.preprocessing_hdfs()
        else:
            self.blocks = self.preprocessing_fs()

        return self

    # TODO:
    def parquet(self, filepath, num_of_parts='*', storage='hdfs',
                host='default', port=0, columns=None):

        if num_of_parts == '*':
            import multiprocessing
            num_of_parts = multiprocessing.cpu_count()

        self.filepath = filepath
        self.nfrag = num_of_parts
        self.storage = storage
        self.host = host
        self.format = 'parquet'
        self.port = port
        self.distributed = self.check_file_or_folder(filepath)

        kwargs = dict()
        kwargs['columns'] = columns
        self.kwargs = kwargs

        if self.storage == 'hdfs':
            self.blocks = self.preprocessing_hdfs()
        else:
            self.blocks = self.preprocessing_fs()

        return self

    def check_file_or_folder(self, fpath):
        if self.storage == 'hdfs':
            from hdfspycompss.hdfs import HDFS
            distributed = HDFS(host=self.host, port=self.port).isdir(fpath)
        else:
            import os
            distributed = os.path.isdir(fpath)
        return distributed

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
                sep = '' if self.filepath[-1] == '/' else '/'
                obj = "{}{}{}".format(self.filepath, sep, name)
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
                             self.dtype, self.error_bad_lines,
                             self.encoding, self.converters,
                             self.parse_dates, 0)
        result, info = parallelize(result, self.nfrag)

        return result, info

    def transform_fs_distributed(self, block, params):

        frag = params['id_frag']
        result, info = _read_fs(block, self.format, self.separator,
                                self.header, self.na_values, self.dtype,
                                self.error_bad_lines, self.encoding,
                                self.converters, self.parse_dates, frag)

        return result, info

    def transform_hdfs(self, block, params):

        frag = params['id_frag']
        result, info = _read_hdfs(block, format_type=self.format,
                                  header=self.header, dtype=self.dtype,
                                  args=self.kwargs, frag=frag)

        return result, info


def _read_fs(filename, format_type, separator, header, na_values,
             dtype, error_bad_lines, encoding, converters, parse_dates, frag):
    """Load a fragment of a csv or json file in a pandas DataFrame."""

    if format_type in ['csv', 'txt']:
        separator = separator[0]
        if header:
            header = 'infer'

        df = pd.read_csv(filename, sep=separator, na_values=na_values,
                         header=header, dtype=dtype[0],
                         error_bad_lines=error_bad_lines, encoding=encoding,
                         converters=converters, parse_dates=parse_dates)

        if not header:
            n_cols = len(df.columns)
            new_columns = ['col_{}'.format(i) for i in range(n_cols)]
            df.columns = new_columns

    else:

        df = pd.read_json(filename, orient='records', dtype=dtype[0],
                          lines=True)

    info = generate_info(df, frag)
    return df, info


def _read_hdfs(blk, format_type, header, dtype, args, frag):
    """Load a DataFrame from a HDFS file."""
    print("[INFO - ReadOperationHDFS] - ", blk)
    from hdfspycompss.block import Block

    df = Block(blk).read_dataframe(format_file=format_type, header=header,
                                   dtype=dtype[0], args=args)

    info = generate_info(df, frag)
    return df, info
