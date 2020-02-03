#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Lucas Miguel S Ponce"
__email__ = "lucasmsp@gmail.com"

from ddf_library.context import COMPSsContext

task_list = None
last_uuid = None


class DataReader(object):

    @staticmethod
    def csv(filepath,
            num_of_parts='*', schema=None, sep=',', header=True, delimiter=None,
            na_filter=True, usecols=None, prefix=None, engine=None,
            converters=None, true_values=None, false_values=None,
            skipinitialspace=False, na_values=None, keep_default_na=True,
            skip_blank_lines=True, parse_dates=False, decimal='.',
            dayfirst=False, thousands=None, quotechar='"', doublequote=True,
            escapechar=None, comment=None, encoding=None, error_bad_lines=True,
            warn_bad_lines=True, delim_whitespace=False, float_precision=None,):

        format_file = 'csv'
        kwargs = locals()
        tmp = _apply_datareader(format_file, kwargs, task_list, last_uuid)
        return tmp

    @staticmethod
    def json(filepath,  num_of_parts='*', schema=None, precise_float=False,
             encoding=None):
        format_file = 'json'
        kwargs = locals()
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
                  num_of_parts='*'):
        """
        Reads a shapefile using the shp and dbf file.

        :param shp_path: Path to the shapefile (.shp)
        :param dbf_path: Path to the shapefile (.dbf)
        :param polygon: Alias to the new column to store the
                polygon coordinates (default, 'points');
        :param attributes: List of attributes to keep in the DataFrame,
                empty to use all fields;
        :param num_of_parts: number of partitions (default, '*' meaning all
         cores available in master CPU);
        :return: DDF

        :Example:

        >>> ddf1 = DDF().load_shapefile(shp_path='/shapefile.shp',
        >>>                             dbf_path='/shapefile.dbf')
        """

        host, port = 'localhost', 9000
        import re
        if re.match(r"hdfs:\/\/+", shp_path):
            storage = 'hdfs'
            host, shp_path = shp_path[7:].split(':')
            port, shp_path = shp_path.split('/', 1)
            shp_path = '/' + shp_path
        elif re.match(r"file:\/\/+", shp_path):
            storage = 'file'
            shp_path = shp_path[7:]
        else:
            raise Exception('`hdfs://` and `file://` storage are supported.')

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

        from ddf_library.functions.geo import read_shapefile

        results, info = read_shapefile(settings, num_of_parts)

        return _submit('load_shapefile', 'other',
                       'COMPLETED', results, info=info)


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
    filepath = kwargs['filepath']
    host, port = 'default', 0
    import re
    if re.match(r"hdfs:\/\/+", filepath):
        storage = 'hdfs'
        host, filename = filepath[7:].split(':')
        port, filename = filename.split('/', 1)
        filename = '/' + filename
        port = int(port)
    elif re.match(r"file:\/\/+", filepath):
        storage = 'file'
        filename = filepath[7:]
    else:
        raise Exception('`hdfs://` and `file://` storage are supported.')

    kwargs['filepath'] = filename
    kwargs['port'] = port
    kwargs['host'] = host
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

            def reader(block, params):
                return data_reader.transform_fs_distributed(block, params)

            COMPSsContext.tasks_map[new_state_uuid] = \
                {'name': 'load_text-file_in',
                 'status': DDF.STATUS_WAIT,
                 'optimization': DDF.OPT_SERIAL,
                 'function': [reader, {}],
                 'output': 1,
                 'input': 0,
                 'parent': [last_uuid]
                 }

        else:

            result, info = data_reader.transform_fs_single()

            COMPSsContext.tasks_map[new_state_uuid] = \
                {'name': 'load_text',
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

        def reader(block, params):
            return data_reader.transform_hdfs(block, params)

        COMPSsContext.tasks_map[new_state_uuid] = \
            {'name': 'load_text-0in',
             'status': DDF.STATUS_WAIT,
             'optimization': DDF.OPT_SERIAL,
             'function': [reader, {}],
             'output': 1,
             'input': 0,
             'parent': [last_uuid]
             }

    return DDF(task_list=task_list.copy(),
               last_uuid=new_state_uuid)
