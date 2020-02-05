#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__ = "Lucas Miguel S Ponce"
__email__ = "lucasmsp@gmail.com"

from pycompss.api.task import task
from pycompss.api.parameter import FILE_IN, FILE_OUT
from pycompss.api.api import compss_open

from ddf_library.utils import read_stage_file


 # TODO: fazer save ser opt_serial

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
        host = settings.get('host', 'default')
        port = settings.get('port', 0)

        if storage == 'hdfs':
            from hdfspycompss.hdfs import HDFS
            dfs = HDFS(host=host, port=port)
            path = "hdfs://"+host+":"+str(port)+"/"+filename
            dfs.mkdir(path)
        else:
            #TODO
            pass

        nfrag = len(data)
        for f in range(nfrag):
            # TODO: create a folder, and then each file

            output = "{}_part_{:05d}".format(filename, f)

            if storage == 'hdfs':
                """Store the DataFrame in CSV, JSON, PICLKE and PARQUET
                 format in HDFS."""
                output = path+"/"+output
                options = [host, port, output, mode, header]
                _save_hdfs_(data[f], options, format_file=format_file)

            elif storage == 'file':
                """Store the DataFrame in CSV, JSON, PICLKE and PARQUET
                 format in common FS."""
                _save_fs_(output, data[f], header, format_file=format_file)
                compss_open(output).close()

        output = {'key_data': ['data'], 'key_info': ['info'],
                  'data': [], 'info': []}
        return output


@task(returns=1, data_input=FILE_IN)
def _save_hdfs_(data_input, settings, format_file='csv'):

    data = read_stage_file(data_input)
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

    success = None
    if format_file == 'csv':
        # 'float_format': '%.6f',
        params_pandas = {'header': header,
                         'index': False, 'sep': ','}
        success = dfs.write_dataframe(filename, data, append=append,
                                      overwrite=over,
                                      params_pandas=params_pandas)
    elif format_file == 'json':
        success = dfs.write_json(filename, data, overwrite=over, append=append)
    elif format_file == 'parquet':
        success = dfs.write_parquet(filename, data, overwrite=over)

    if not success:
        raise Exception('Error in SaveHDFSOperation.')
    return []


@task(filename=FILE_OUT, data_input=FILE_IN)
def _save_fs_(filename, data_input, header, format_file):
    """
    Method used to save a DataFrame into a file.

    :param filename: The name used in the output.
    :param data: The pandas DataFrame which you want to save.
    """
    data = read_stage_file(data_input)
    mode = 'w'
    if format_file == 'csv':
        if header:
            data.to_csv(filename, sep=',', mode=mode, header=True, index=False)
        else:
            data.to_csv(filename, sep=',', mode=mode, header=False, index=False)
    elif format_file == 'json':
        data.to_json(filename, orient='records')
    elif format_file == 'pickle':
        import _pickle as pickle
        with open(filename, 'wb') as output:
            pickle.dump(data, output, pickle.HIGHEST_PROTOCOL)
    elif format_file == 'parquet':
        data.to_parquet(filename)
    else:
        raise Exception("FORMAT NOT SUPPORTED!")
