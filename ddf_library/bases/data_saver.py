#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Lucas Miguel S Ponce"
__email__ = "lucasmsp@gmail.com"

from ddf_library.bases.metadata import OPTGroup, Status
from ddf_library.bases.context_base import ContextBase
from ddf_library.functions.etl.save_data import DataSaver
from ddf_library.utils import parser_filepath

last_uuid = None


class Save(object):
    # noinspection PyUnresolvedReferences

    @staticmethod
    def csv(filepath, header=True, mode=DataSaver.MODE_OVERWRITE, sep=',',
            na_rep='', float_format=None, columns=None, encoding=None,
            quoting=None, quotechar='"', date_format=None, doublequote=True,
            escapechar=None, decimal='.'):
        """
        Saves a csv file.

        :param filepath:
        :param header:
        :param mode:
        :param sep:
        :param na_rep:
        :param float_format:
        :param columns:
        :param encoding:
        :param quoting:
        :param quotechar:
        :param date_format:
        :param doublequote:
        :param escapechar:
        :param decimal:
        :return:
        """

        format_file = DataSaver.FORMAT_CSV
        kwargs = locals()
        _apply_datasaver(format_file, kwargs, last_uuid)
        return None

    @staticmethod
    def json(filepath, mode=DataSaver.MODE_OVERWRITE, date_format=None,
             double_precision=10, force_ascii=True, date_unit='ms'):
        """
        Saves a json file.

        :param filepath:
        :param mode:
        :param date_format:
        :param double_precision:
        :param force_ascii:
        :param date_unit:
        :return:
        """
        format_file = DataSaver.FORMAT_JSON
        kwargs = locals()
        _apply_datasaver(format_file, kwargs, last_uuid)
        return None

    @staticmethod
    def parquet(filepath, mode=DataSaver.MODE_OVERWRITE, compression='snappy'):
        """
        Saves a parquet file.

        :param filepath:
        :param mode:
        :param compression:
        :return:
        """
        format_file = DataSaver.FORMAT_PARQUET
        kwargs = locals()
        _apply_datasaver(format_file, kwargs, last_uuid)
        return None

    @staticmethod
    def pickle(filepath, mode=DataSaver.MODE_OVERWRITE, compression='infer'):
        """
        Saves a pickle file.

        :param filepath:
        :param mode:
        :param compression:
        :return:
        """
        format_file = DataSaver.FORMAT_PICKLE
        kwargs = locals()
        _apply_datasaver(format_file, kwargs, last_uuid)
        return None


def _apply_datasaver(format_file, kwargs, uuid):
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

        from ddf_library.bases.optimizer.operations import DataWriter

        new_state_uuid = ContextBase \
            .ddf_add_task(operation=DataWriter(data_saver, settings,
                                               tag="save-"+storage),
                          parent=[uuid])

        tmp = DDF(last_uuid=new_state_uuid)
        tmp.last_uuid = ContextBase().run_workflow(tmp.last_uuid)
