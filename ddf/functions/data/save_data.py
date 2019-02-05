#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__ = "Lucas Miguel S Ponce"
__email__ = "lucasmsp@gmail.com"

import pandas as pd
from pycompss.api.task import task
from pycompss.api.parameter import FILE_OUT


class SaveOperation(object):

    def preprocessing(self, settings, nfrag):
        if any(['format' not in settings,
                'filename' not in settings]):
            raise \
                Exception('SaveOperation: Please inform filename and format.')

        if settings.get('storage', 'hdfs') == 'hdfs':
            from hdfspycompss.Utils import get_suffixes
            settings['suffixes'] = get_suffixes(nfrag)
        return settings

    def transform(self, data, settings, nfrag):

        settings = self.preprocessing(settings, nfrag)

        storage = settings.get('storage', 'hdfs')
        format_file = settings['format']
        filename = settings['filename']
        output = [[] for _ in range(nfrag)]

        if storage == 'hdfs':
            """Store the dataFrame in csv or json format in HDFS."""
            host = settings.get('host', 'localhost')
            port = settings.get('port', 9000)
            mode = settings['mode']

            suffixes = settings['suffixes']

            if format_file == 'csv':

                header = settings['header']
                for i, f in enumerate(suffixes):
                    name = "{}_part_{}".format(filename, f)
                    output[i] = _save_csv_hdfs(host, port, name,
                                               data[i], mode, header)

            elif format_file == 'json':
                for i, f in enumerate(suffixes):
                    name = "{}_part_{}".format(filename, f)
                    output[i] = _save_json_hdfs(host, port,
                                                name, data[i], mode)

        elif storage == 'fs':
            """Store the dataFrame in CSV or JSON format in common FS."""

            if format_file == 'CSV':
                mode = settings['mode']
                header = settings['header']
                for f in range(nfrag):
                    output[f] = "{}_part_{}".format(filename, f)
                    _save_csv_fs(output, data[f], mode, header)

            elif format_file == 'JSON':
                for f in range(nfrag):
                    output[f] = "{}_part_{}".format(filename, f)
                    _save_json_fs(output, data[f])
        return data

    def transform_serial(self, data, settings):
        storage = settings.get('storage', 'hdfs')
        format_file = settings['format']
        filename = settings['filename']
        idfrag = settings['id_frag']

        if storage == 'hdfs':
            """Store the dataFrame in CSV or JSON format in HDFS."""
            host = settings.get('host', 'localhost')
            port = settings.get('port', 9000)
            mode = settings['mode']
            suffixes = settings['suffixes']

            if format_file == 'csv':
                header = settings['header']
                output = "{}_part_{}".format(filename, suffixes[idfrag])
                _save_csv_hdfs_(host, port, output, data, mode, header)

            elif format_file == 'json':
                output = "{}_part_{}".format(filename, suffixes[idfrag])
                _save_json_fs_(host, port, output, data, mode)

        elif storage == 'fs':
            """Store the dataFrame in CSV or JSON format in common FS."""

            if format_file == 'csv':
                mode = settings['mode']
                header = settings['header']
                output = "{}_part_{}".format(filename, idfrag)
                _save_csv_fs_(output, data, mode, header)

            elif format_file == 'json':
                output = "{}_part_{}".format(filename, idfrag)
                _save_json_fs_(output, data)

        return data


@task(returns=list)
def _save_csv_hdfs(host, port, filename, data, mode, header):
    return _save_csv_hdfs_(host, port, filename, data, mode, header)


@task(filename=FILE_OUT)
def _save_csv_fs(filename, data, mode, header):
        _save_csv_fs_(filename, data, mode, header)


@task(returns=list)
def _save_json_hdfs(host, port, filename, data, mode):
    return _save_json_hdfs_(host, port, filename, data, mode)


@task(filename=FILE_OUT)
def _save_json_fs(filename, data):
        _save_json_fs_(filename, data)


def _save_csv_hdfs_(host, port, filename, data, mode, header):
    from hdfspycompss.HDFS import HDFS
    dfs = HDFS(host=host, port=port)
    over = False
    append = False

    if mode == 'overwrite':
        over = True
    elif mode == 'append':
        append = True
    elif dfs.exist(filename):
        if mode == 'ignore':
            return []
        elif mode == 'error':
            raise Exception('SaveOperation: File already exists.')

    success = dfs.writeDataFrame(filename, data, header=header,
                                 overwrite=over, append=append)
    if not success:
        raise Exception('Error in SaveHDFSOperation.')
    return filename


def _save_csv_fs_(filename, data, mode, header):
    """Method used to save a dataFrame into a file (CSV).

    :param filename: The name used in the output.
    :param data: The pandas dataFrame which you want to save.
    :param mode: append, overwrite, ignore or error
    """
    import os.path

    if mode is 'append':
        mode = 'a'
    elif os.path.exists(filename):
        if mode is 'ignore':
            return None
        elif mode is 'error':
            raise Exception('SaveDataOperation: File already exists.')
    else:
        mode = 'w'

    if len(data) == 0:
        data = pd.DataFrame()
    if header:
        data.to_csv(filename, sep=',', mode=mode, header=True, index=False)
    else:
        data.to_csv(filename, sep=',', mode=mode, header=False, index=False)

    return None


def _save_json_hdfs_(host, port, filename, data, mode):
    """Method used to save a dataFrame into a JSON (following the 'records'
    pandas orientation).

    :param filename: The name used in the output.
    :param data: The pandas dataframe which you want to save.
    """
    from hdfspycompss.HDFS import HDFS
    dfs = HDFS(host=host, port=port)
    over = False
    append = False

    if mode is 'overwrite':
        over = True
    elif mode is 'append':
        append = True
    elif dfs.exist(filename):
        if mode is 'ignore':
            return []
        elif mode is 'error':
            raise Exception('SaveOperation: File already exists.')

    sucess = dfs.writeJson(filename, data, overwrite=over, append=append)
    return filename


def _save_json_fs_(filename, data):
    """Method used to save a dataFrame into a JSON (following the 'records'
    pandas orientation).

    :param filename: The name used in the output.
    :param data: The pandas dataframe which you want to save.
    """
    data.to_json(filename, orient='records')
    return None


# def save_pickle_fs(outfile, data):
#     """Save an array to a serializable Pickle file format
#
#     :param outfile: the /path/file.npy
#     :param data: the data to save
#     """
#     with open(outfile, 'wb') as handle:
#         pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)