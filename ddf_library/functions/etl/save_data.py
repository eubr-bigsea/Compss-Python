#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__ = "Lucas Miguel S Ponce"
__email__ = "lucasmsp@gmail.com"

from ddf_library.utils import generate_info

from pycompss.api.task import task
from pycompss.api.parameter import FILE_OUT

import pandas as pd


class SaveOperation(object):
    """Save DataFrame as json or csv format in HDFS or common file system."""

    @staticmethod
    def preprocessing(settings):
        if any(['format' not in settings,
                'filename' not in settings]):
            raise \
                Exception('SaveOperation: Please inform filename and format.')

    @staticmethod
    def transform(data, settings):

        format_file = settings['format']
        filename = settings['filename']
        mode = settings['mode']
        header = settings['header']

        frag = settings['id_frag']

        storage = settings.get('storage', 'hdfs')
        host = settings.get('host', 'localhost')
        port = settings.get('port', 9000)

        output = "{}_part_{}".format(filename, frag)

        if storage == 'hdfs':
            """Store the DataFrame in CSV or JSON format in HDFS."""

            if format_file == 'csv':
                _save_csv_hdfs_(host, port, output, data, mode, header)

            elif format_file == 'json':
                _save_json_hdfs_(host, port, output, data, mode)

        elif storage == 'fs':
            """Store the DataFrame in CSV or JSON format in common FS."""

            if format_file == 'csv':
                _save_csv_fs_(output, data, header)

            elif format_file == 'json':
                _save_json_fs_(output, data)

        info = generate_info(data, frag)
        return data, info


def _save_csv_hdfs_(host, port, filename, data, mode, header):
    from hdfspycompss.hdfs import HDFS
    dfs = HDFS(host=host, port=port)
    over, append = False,  False

    if mode == 'overwrite':
        over = True
    elif mode == 'append':
        append = True
    elif dfs.exist(filename):
        if mode == 'ignore':
            return None
        elif mode == 'error':
            raise Exception('SaveOperation: File already exists.')

    success = dfs.write_dataframe(filename, data, header=header,
                                  overwrite=over, append=append)
    if not success:
        raise Exception('Error in SaveHDFSOperation.')


def _save_json_hdfs_(host, port, filename, data, mode):
    """
    Method used to save a DataFrame into a JSON (using the 'records'
    pandas orientation).

    :param filename: The name used in the output.
    :param data: The pandas DataFrame which you want to save.
    """
    from hdfspycompss.hdfs import HDFS
    dfs = HDFS(host=host, port=port)
    over, append = False,  False

    if mode is 'overwrite':
        over = True
    elif mode is 'append':
        append = True
    elif dfs.exist(filename):
        if mode is 'ignore':
            return None
        elif mode is 'error':
            raise Exception('SaveOperation: File already exists.')

    success = dfs.write_json(filename, data, overwrite=over, append=append)

    if not success:
        raise Exception('Error in SaveHDFSOperation.')


def _save_csv_fs_(filename, data, header):
    """
    Method used to save a DataFrame into a file (CSV).

    :param filename: The name used in the output.
    :param data: The pandas DataFrame which you want to save.
    """

    mode = 'w'

    if header:
        data.to_csv(filename, sep=',', mode=mode, header=True, index=False)
    else:
        data.to_csv(filename, sep=',', mode=mode, header=False, index=False)


def _save_json_fs_(filename, data):
    """
    Method used to save a dataFrame into a JSON (using the 'records'
    pandas orientation).

    :param filename: The name used in the output.
    :param data: The pandas DataFrame which you want to save.
    """
    data.to_json(filename, orient='records')

