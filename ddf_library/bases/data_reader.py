#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Lucas Miguel S Ponce"
__email__ = "lucasmsp@gmail.com"

from ddf_library.context import COMPSsContext
from ddf_library.utils import parser_filepath

task_list = None
last_uuid = None


class DataReader(object):

    @staticmethod
    def csv(filepath,
            num_of_parts='*', schema='infer', sep=',', header=True,
            delimiter=None, na_filter=True, usecols=None, prefix=None,
            engine=None, converters=None, true_values=None, false_values=None,
            skipinitialspace=False, na_values=None, keep_default_na=True,
            skip_blank_lines=True, parse_dates=False, decimal='.',
            dayfirst=False, thousands=None, quotechar='"', doublequote=True,
            escapechar=None, comment=None, encoding='utf-8',
            error_bad_lines=True, warn_bad_lines=True, delim_whitespace=False,
            float_precision=None):

        format_file = 'csv'
        kwargs = locals()
        kwargs['schema'] = _check_schema(kwargs['schema'])
        tmp = _apply_datareader(format_file, kwargs, task_list, last_uuid)
        return tmp

    @staticmethod
    def json(filepath,  num_of_parts='*', schema='infer', precise_float=False,
             encoding='utf-8'):
        format_file = 'json'
        kwargs = locals()
        kwargs['schema'] = _check_schema(kwargs['schema'])
        tmp = _apply_datareader(format_file, kwargs, task_list, last_uuid)
        return tmp

    @staticmethod
    def parquet(filepath, num_of_parts='*', columns=None):
        format_file = 'parquet'
        kwargs = locals()
        tmp = _apply_datareader(format_file, kwargs, task_list, last_uuid)
        return tmp

    @staticmethod
    def shapefile(shp_path, dbf_path, polygon='points', attributes=None,
                  num_of_parts='*', schema='infer'):
        """
        Reads a shapefile using the shp and dbf file.

        :param shp_path: Path to the shapefile (.shp)
        :param dbf_path: Path to the shapefile (.dbf)
        :param polygon: Alias to the new column to store the
                polygon coordinates (default, 'points');
        :param attributes: List of attributes to keep in the DataFrame,
                empty to use all fields;
        :param schema: 'infer' to infer schema, otherwise, provide the dtype
        :param num_of_parts: number of partitions (default, '*' meaning all
         cores available in master CPU);
        :return: DDF

        :Example:

        >>> ddf1 = DDF().load_shapefile(shp_path='/shapefile.shp',
        >>>                             dbf_path='/shapefile.dbf')
        """

        host, port, shp_path, storage = parser_filepath(shp_path)
        _, _, dbf_path, _ = parser_filepath(dbf_path)

        if attributes is None:
            attributes = []

        if isinstance(num_of_parts, str):
            import multiprocessing
            num_of_parts = multiprocessing.cpu_count()

        settings = dict()
        settings['shp_path'] = shp_path
        settings['dbf_path'] = dbf_path
        settings['polygon'] = polygon
        settings['attributes'] = attributes
        settings['host'] = host
        settings['port'] = int(port)
        settings['storage'] = storage
        settings['schema'] = _check_schema(schema)

        from ddf_library.functions.geo import read_shapefile

        results, info = read_shapefile(settings, num_of_parts)

        return _submit('load_shapefile', 'other',
                       'COMPLETED', results, info=info)


def _check_schema(schema):
    from ddf_library.types import _converted_types
    if isinstance(schema, dict):
        for key in schema:
            t = schema[key]
            if t not in _converted_types:
                raise Exception("Type is not supported.")
            else:
                schema[key] = _converted_types[t]
    elif isinstance(schema, str):
        if schema == 'infer':
            schema = 'infer'
        elif schema not in _converted_types:
            raise Exception("Type is not supported.")
        else:
            schema = _converted_types[schema]
    else:
        raise Exception("schema must be a string or a dictionary.")
    return schema


def _submit(name_task, optimization, status, results, info=False):
    from ddf_library.ddf import DDF
    new_state_uuid = DDF._generate_uuid()
    if info:
        COMPSsContext.catalog[new_state_uuid] = info
    COMPSsContext.tasks_map[new_state_uuid] = \
        {'name': name_task,
         'status': status,
         'optimization': optimization,
         'function': None,
         'result': results,
         'output': 1, 'input': 0,
         'parent': [last_uuid]
         }

    return DDF(task_list=task_list, last_uuid=new_state_uuid)


def _apply_datareader(format_file, kwargs, task_list, last_uuid):
    from ddf_library.ddf import DDF

    host, port, filename, storage = parser_filepath(kwargs['filepath'])

    kwargs['filepath'] = filename
    kwargs['port'] = port
    kwargs['host'] = host
    kwargs['storage'] = storage
    kwargs.pop('format_file', None)

    from ddf_library.functions.etl.read_data import DataReader
    if format_file == 'csv':
        data_reader = DataReader().csv(**kwargs)
    elif format_file == 'json':
        data_reader = DataReader().json(**kwargs)
    else:
        raise Exception('File formart not supported.')

    new_state_uuid = DDF._generate_uuid()
    if storage is 'file':

        if data_reader.distributed:
            # setting the last task's input (init)
            blocks = data_reader.get_blocks()
            COMPSsContext.tasks_map[last_uuid]['result'] = blocks

            def task_read_many_fs(block, params):
                return data_reader.transform_fs_distributed(block, params)

            COMPSsContext.tasks_map[new_state_uuid] = \
                {'name': 'read-many-fs',
                 'status': DDF.STATUS_WAIT,
                 'optimization': DDF.OPT_SERIAL,
                 'function': [task_read_many_fs, {}],
                 'output': 1,
                 'input': 0,
                 'parent': [last_uuid]
                 }

        else:

            result, info = data_reader.transform_fs_single()

            COMPSsContext.tasks_map[new_state_uuid] = \
                {'name': 'read-one-fs',
                 'status': DDF.STATUS_COMPLETED,
                 'optimization': DDF.OPT_OTHER,
                 'function': None,
                 'result': result,
                 'output': 1,
                 'input': 0,
                 'parent': [last_uuid]
                 }

            COMPSsContext.catalog[new_state_uuid] = info
    else:
        blocks = data_reader.get_blocks()

        COMPSsContext.tasks_map[last_uuid]['result'] = blocks

        def task_read_hdfs(block, params):
            return data_reader.transform_hdfs(block, params)

        COMPSsContext.tasks_map[new_state_uuid] = \
            {'name': 'read-hdfs',
             'status': DDF.STATUS_WAIT,
             'optimization': DDF.OPT_SERIAL,
             'function': [task_read_hdfs, {}],
             'output': 1,
             'input': 0,
             'parent': [last_uuid]
             }

    return DDF(task_list=task_list.copy(),
               last_uuid=new_state_uuid)
