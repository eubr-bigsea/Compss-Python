#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__ = "Lucas Miguel S Ponce"
__email__ = "lucasmsp@gmail.com"

from pycompss.api.task import task
from pycompss.api.parameter import FILE_OUT


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

        storage = settings.get('storage', 'hdfs')
        host = settings.get('host', 'localhost')
        port = settings.get('port', 9000)

        nfrag = len(data)
        for f in range(nfrag):

            output = "{}_part_{}".format(filename, f)

            if storage == 'hdfs':
                """Store the DataFrame in CSV or JSON format in HDFS."""

                if format_file == 'csv':
                    options = [host, port, output, mode, header]
                    _save_csv_hdfs_(data[f], options)

                elif format_file == 'json':
                    options = [host, port, output, mode]
                    _save_json_hdfs_(data[f], options)

            elif storage == 'fs':
                """Store the DataFrame in CSV or JSON format in common FS."""

                if format_file == 'csv':
                    _save_csv_fs_(output, data[f], header)

                elif format_file == 'json':
                    _save_json_fs_(output, data[f])

                elif format_file == 'pickle':
                    _save_pickle_fs_(output, data[f])

        output = {'key_data': ['data'], 'key_info': ['info'],
                  'data': [], 'info': []}
        return output


@task(returns=1)
def _save_csv_hdfs_(data, settings):
    host, port, filename, mode, header = settings
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

    # 'float_format': '%.6f',
    params_pandas = {'header': header,
                     'index': False, 'sep': ','}
    success = dfs.write_dataframe(filename, data, append=append,
                                  overwrite=over,
                                  params_pandas=params_pandas)
    if not success:
        raise Exception('Error in SaveHDFSOperation.')
    return []


@task(returns=1)
def _save_json_hdfs_(data, settings):
    """
    Method used to save a DataFrame into a JSON (using the 'records'
    pandas orientation).

    :param filename: The name used in the output.
    :param data: The pandas DataFrame which you want to save.
    """
    host, port, filename, mode = settings

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
    return []


@task(filename=FILE_OUT)
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


@task(filename=FILE_OUT)
def _save_json_fs_(filename, data):
    """
    Method used to save a dataFrame into a JSON (using the 'records'
    pandas orientation).

    :param filename: The name used in the output.
    :param data: The pandas DataFrame which you want to save.
    """
    data.to_json(filename, orient='records')


@task(filename=FILE_OUT)
def _save_pickle_fs_(filename, data):
    """
    Method used to save a dataFrame into a Pickle.

    :param filename: The name used in the output.
    :param data: The pandas DataFrame which you want to save.
    """
    import _pickle as pickle
    with open(filename, 'wb') as output:
        pickle.dump(data, output, pickle.HIGHEST_PROTOCOL)
