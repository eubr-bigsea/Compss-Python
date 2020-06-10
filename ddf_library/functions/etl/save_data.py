#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__ = "Lucas Miguel S Ponce"
__email__ = "lucasmsp@gmail.com"

from ddf_library.utils import generate_info


class DataSaver(object):
    MODE_OVERWRITE = 'overwrite'
    MODE_IGNORE = 'ignore'
    MODE_ERROR = 'error'

    FORMAT_CSV = 'csv'
    FORMAT_JSON = 'json'
    FORMAT_PARQUET = 'parquet'
    FORMAT_PICKLE = 'pickle'

    STORAGE_HDFS = 'hdfs'
    STORAGE_FS = 'file'

    def __init__(self):
        self.format = None
        self.storage = None
        self.mode = None
        self.host = None
        self.port = None
        self.filepath = None
        self.kwargs = None

    def prepare_csv(self, filepath, mode=MODE_OVERWRITE, storage=STORAGE_HDFS,
                    host='default', port=0, header=True, sep=',', na_rep='',
                    float_format=None, columns=None, encoding=None,
                    quoting=None, quotechar='"', date_format=None,
                    doublequote=True, escapechar=None, decimal='.'):

        self.filepath = filepath
        self.mode = mode
        self.storage = storage
        self.host = host
        self.port = int(port)
        self.format = self.FORMAT_CSV

        kwargs = dict()
        kwargs['header'] = header
        kwargs['sep'] = sep
        kwargs['na_rep'] = na_rep
        kwargs['float_format'] = float_format
        kwargs['columns'] = columns
        kwargs['encoding'] = encoding
        kwargs['quoting'] = quoting
        kwargs['quotechar'] = quotechar
        kwargs['date_format'] = date_format
        kwargs['doublequote'] = doublequote
        kwargs['escapechar'] = escapechar
        kwargs['decimal'] = decimal
        self.kwargs = kwargs
        return self

    def prepare_json(self, filepath, mode=MODE_OVERWRITE, storage='fs',
                     host='default', port=0, date_format=None,
                     double_precision=10, force_ascii=True, date_unit='ms'):

        self.filepath = filepath
        self.mode = mode
        self.storage = storage
        self.host = host
        self.port = int(port)
        self.format = self.FORMAT_JSON

        kwargs = dict()
        kwargs['double_precision'] = double_precision
        kwargs['date_format'] = date_format
        kwargs['force_ascii'] = force_ascii
        kwargs['date_unit'] = date_unit
        self.kwargs = kwargs
        return self

    def prepare_parquet(self, filepath, mode=MODE_OVERWRITE, storage='fs',
                        host='default', port=0, compression='snappy'):

        self.filepath = filepath
        self.mode = mode
        self.storage = storage
        self.host = host
        self.port = int(port)
        self.format = self.FORMAT_PARQUET

        self.kwargs = {'compression': compression}
        return self

    def prepare_pickle(self, filepath, mode=MODE_OVERWRITE, storage='fs',
                       host='default', port=0, compression='infer',
                       protocol=-1):

        self.filepath = filepath
        self.mode = mode
        self.storage = storage
        self.host = host
        self.port = int(port)
        self.format = self.FORMAT_PARQUET

        self.kwargs = {'compression': compression, 'protocol': protocol}
        return self

    def check_path(self):

        if self.storage == self.STORAGE_HDFS:
            from hdfspycompss.hdfs import HDFS
            dfs = HDFS(host=self.host, port=self.port)
            path = 'hdfs://{}:{}{}'.format(self.host, self.port, self.filepath)
            if dfs.exist(self.filepath):
                if self.mode == self.MODE_ERROR:
                    raise Exception('Filepath {} already exists.'.format(path))
                elif self.mode == self.MODE_IGNORE:
                    return 'ignored'
                else:
                    dfs.rm(self.filepath, recursive=True)
            dfs.mkdir(self.filepath)

        elif self.storage == self.STORAGE_FS:
            import os
            from pathlib import Path
            path = 'file://{}'.format(self.filepath)
            if os.path.exists(self.filepath):
                if self.mode == self.MODE_ERROR:
                    raise Exception('Filepath {} already exists.'.format(path))
                elif self.mode == self.MODE_IGNORE:
                    return 'ignored'
                else:
                    if os.path.isfile(self.filepath):
                        os.remove(self.filepath)
            Path(self.filepath).mkdir(parents=True, exist_ok=True)
        else:
            raise Exception('Storage not supported.')
        return 'ok'

    def generate_names(self, nfrag):
        output = ['{}/output_part_{:05d}.{}'.format(self.filepath, i,
                                                    self.format)
                  for i in range(nfrag)]
        return output

    def save(self, df, settings):
        output = settings['output'][settings['id_frag']]

        if self.storage == self.STORAGE_HDFS:
            """Store the DataFrame in CSV, JSON, PICLKE and PARQUET
             format in HDFS."""
            from hdfspycompss.hdfs import HDFS
            dfs = HDFS(host=self.host, port=self.port)

            if self.format == self.FORMAT_CSV:
                self.kwargs['index'] = False
                dfs.write_pandas_to_csv(output, df, **self.kwargs)
            elif self.format == self.FORMAT_JSON:
                dfs.write_pandas_to_json(output, df, **self.kwargs)
            elif self.format == self.FORMAT_PARQUET:
                dfs.write_pandas_to_parquet(output, df, **self.kwargs)
            elif self.format == self.FORMAT_PICKLE:
                dfs.write_pandas_to_pickle(output, df, **self.kwargs)

        elif self.storage == self.STORAGE_FS:
            """Store the DataFrame in CSV, JSON, PICLKE and PARQUET
             format in common FS."""
            if self.format == self.FORMAT_CSV:
                df.to_csv(output, index=False, **self.kwargs)
            elif self.format == self.FORMAT_JSON:
                df.to_json(output, orient='records', **self.kwargs)
            elif self.format == self.FORMAT_PARQUET:
                df.to_paquet(output, **self.kwargs)
            elif self.format == self.FORMAT_PICKLE:
                df.to_picke(output, **self.kwargs)
            else:
                raise Exception("FORMAT NOT SUPPORTED!")

        info = generate_info(df, settings['id_frag'])
        return df, info
