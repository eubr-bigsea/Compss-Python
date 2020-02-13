#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Lucas Miguel S Ponce"
__email__ = "lucasmsp@gmail.com"

from ddf_library.context import COMPSsContext
from ddf_library.functions.etl.save_data import DataSaver
from ddf_library.utils import parser_filepath

task_list = None
last_uuid = None


class Save(object):
    # noinspection PyUnresolvedReferences
    """
    Save the data in the storage.

    :param filepath: output file path;
    :param format: format file, csv, json or a pickle;
    :param header: save with the columns header;
    :param mode: 'overwrite' (default) if file exists, 'ignore' or 'error'.
     Only used when storage is 'hdfs'.
    :return: Return the same input data to perform others operations;
    """

    @staticmethod
    def csv(filepath, header=True, mode=DataSaver.MODE_OVERWRITE, sep=',',
            na_rep='', float_format=None, columns=None, encoding=None,
            quoting=None, quotechar='"', date_format=None, doublequote=True,
            escapechar=None, decimal='.'):

        format_file = DataSaver.FORMAT_CSV
        kwargs = locals()
        _apply_datasaver(format_file, kwargs, task_list, last_uuid)
        return None

    @staticmethod
    def json(filepath, mode=DataSaver.MODE_OVERWRITE, date_format=None,
             double_precision=10, force_ascii=True, date_unit='ms'):
        format_file = DataSaver.FORMAT_JSON
        kwargs = locals()
        _apply_datasaver(format_file, kwargs, task_list, last_uuid)
        return None

    @staticmethod
    def parquet(filepath, mode=DataSaver.MODE_OVERWRITE, compression='snappy'):
        format_file = DataSaver.FORMAT_PARQUET
        kwargs = locals()
        _apply_datasaver(format_file, kwargs, task_list, last_uuid)
        return None

    @staticmethod
    def pickle(filepath, mode=DataSaver.MODE_OVERWRITE, compression='infer'):
        format_file = DataSaver.FORMAT_PICKLE
        kwargs = locals()
        _apply_datasaver(format_file, kwargs, task_list, last_uuid)
        return None


def _apply_datasaver(format_file, kwargs, task_list, last_uuid):
    from ddf_library.ddf import DDF
    host, port, filename, storage = parser_filepath(kwargs['filepath'])

    kwargs['filepath'] = filename
    kwargs['port'] = int(port)
    kwargs['host'] = host
    kwargs['storage'] = storage
    kwargs.pop('format_file', None)

    if format_file == DataSaver.FORMAT_CSV:
        data_saver = DataSaver().prepare_csv(**kwargs)
    elif format_file == DataSaver.FORMAT_JSON:
        data_saver = DataSaver().prepare_json(**kwargs)
    elif format_file == DataSaver.FORMAT_PARQUET:
        data_saver = DataSaver().prepare_parquet(**kwargs)
    elif format_file == DataSaver.FORMAT_PICKLE:
        data_saver = DataSaver().prepare_pickle(**kwargs)
    else:
        raise Exception('File format not supported.')

    status_path = data_saver.check_path()
    if status_path == 'ok':
        settings = {'output': data_saver.generate_names}

        def task_save(df, params):
            return data_saver.save(df, params)

        new_state_uuid = DDF._generate_uuid()
        COMPSsContext.catalog_tasks[new_state_uuid] = \
            {'name': 'save-{}'.format(storage),
             'status': DDF.STATUS_WAIT,
             'optimization': DDF.OPT_SERIAL,
             'function': [task_save, settings],
             'output': 0, 'input': 1,
             'parent': [last_uuid]
             }

        DDF(task_list=task_list.copy(), last_uuid=new_state_uuid)\
            ._run_compss_context()
