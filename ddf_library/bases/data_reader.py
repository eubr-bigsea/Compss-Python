#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Lucas Miguel S Ponce"
__email__ = "lucasmsp@gmail.com"

from ddf_library.utils import parser_filepath
from ddf_library.bases.metadata import Status, OPTGroup


class DataReader(object):

    @staticmethod
    def csv(filepath,
            num_of_parts='*', schema='str', sep=',', header=True,
            delimiter=None, na_filter=True, usecols=None, prefix=None,
            engine=None, converters=None, true_values=None, false_values=None,
            skipinitialspace=False, na_values=None, keep_default_na=True,
            skip_blank_lines=True, parse_dates=False, decimal='.',
            dayfirst=False, thousands=None, quotechar='"', doublequote=True,
            escapechar=None, comment=None, encoding='utf-8',
            error_bad_lines=True, warn_bad_lines=True, delim_whitespace=False,
            float_precision=None):
        """
        Reads a csv file.

        :param filepath:
        :param num_of_parts:
        :param schema:
        :param sep:
        :param header:
        :param delimiter:
        :param na_filter:
        :param usecols:
        :param prefix:
        :param engine:
        :param converters:
        :param true_values:
        :param false_values:
        :param skipinitialspace:
        :param na_values:
        :param keep_default_na:
        :param skip_blank_lines:
        :param parse_dates:
        :param decimal:
        :param dayfirst:
        :param thousands:
        :param quotechar:
        :param doublequote:
        :param escapechar:
        :param comment:
        :param encoding:
        :param error_bad_lines:
        :param warn_bad_lines:
        :param delim_whitespace:
        :param float_precision:
        :return:
        """

        format_file = 'csv'
        kwargs = locals()
        kwargs['schema'] = _check_schema(kwargs['schema'])
        tmp = _apply_datareader(format_file, kwargs)
        return tmp

    @staticmethod
    def json(filepath,  num_of_parts='*', schema='str', precise_float=False,
             encoding='utf-8'):
        """
        Reads a json file.

        :param filepath:
        :param num_of_parts:
        :param schema:
        :param precise_float:
        :param encoding:
        :return:
        """
        format_file = 'json'
        kwargs = locals()
        kwargs['schema'] = _check_schema(kwargs['schema'])
        tmp = _apply_datareader(format_file, kwargs)
        return tmp

    @staticmethod
    def parquet(filepath, num_of_parts='*', columns=None):
        """
        Reads a parquet file.

        :param filepath:
        :param num_of_parts:
        :param columns:
        :return:
        """
        format_file = 'parquet'
        kwargs = locals()
        tmp = _apply_datareader(format_file, kwargs)
        return tmp

    @staticmethod
    def shapefile(shp_path, dbf_path, polygon='points', attributes=None,
                  num_of_parts='*', schema='str'):
        # noinspection PyUnresolvedReferences
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

        >>> ddf1 = COMPSsContext()\
        >>> .read.shapefile(shp_path='hdfs://localhost:9000/shapefile.shp',
        >>>                 dbf_path='hdfs://localhost:9000/shapefile.dbf')
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
        settings['nfrag'] = num_of_parts

        from ddf_library.bases.context_base import ContextBase
        from ddf_library.bases.optimizer.operations import ReadShapefile
        from ddf_library.ddf import DDF

        if storage == 'hdfs':

            new_state_uuid = ContextBase\
                .ddf_add_task(operation=ReadShapefile(settings),
                              parent=[])
            return DDF(last_uuid=new_state_uuid)

        else:
            from ddf_library.functions.geo import read_shapefile_all
            result, info = read_shapefile_all(settings, num_of_parts)

            new_state_uuid = ContextBase \
                .ddf_add_task(operation=ReadShapefile({},
                                                      opt=OPTGroup.OPT_OTHER),
                              status=Status.STATUS_COMPLETED,
                              result=result,
                              info_data=info)

            return DDF(last_uuid=new_state_uuid)


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
        if schema == 'str':
            schema = 'str'
        elif schema not in _converted_types:
            raise Exception("Type is not supported.")
        else:
            schema = _converted_types[schema]
    else:
        raise Exception("schema must be a string or a dictionary.")
    return schema


def _apply_datareader(format_file, kwargs):

    host, port, filename, storage = parser_filepath(kwargs['filepath'])

    kwargs['filepath'] = filename
    kwargs['port'] = port
    kwargs['host'] = host
    kwargs['storage'] = storage
    kwargs.pop('format_file', None)

    from ddf_library.functions.etl.read_data import DataReader
    from ddf_library.bases.context_base import ContextBase
    from ddf_library.ddf import DDF

    if format_file == 'csv':
        data_reader = DataReader().csv(**kwargs)
    elif format_file == 'json':
        data_reader = DataReader().json(**kwargs)
    else:
        raise Exception('File format not supported.')

    from ddf_library.bases.optimizer.operations import DataReaderOperation

    if storage is 'file':
        if data_reader.distributed:
            op = DataReaderOperation(data_reader=data_reader, settings={},
                                     opt=OPTGroup.OPT_SERIAL,
                                     tag='read-many-file')

        else:
            op = DataReaderOperation(data_reader=data_reader, settings={},
                                     opt=OPTGroup.OPT_OTHER,
                                     tag='read-one-file')

    else:
        op = DataReaderOperation(data_reader=data_reader, settings={},
                                 opt=OPTGroup.OPT_SERIAL,
                                 tag='read-hdfs')

    new_state_uuid = ContextBase.ddf_add_task(operation=op, parent=[])

    return DDF(last_uuid=new_state_uuid)
