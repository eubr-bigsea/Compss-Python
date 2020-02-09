#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Lucas Miguel S Ponce"
__email__ = "lucasmsp@gmail.com"


__all__ = ['col', 'udf']

import datetime

from ddf_library.types import _converted_types


class udf(object):

    def __init__(self, function, type, *args):
        self.function = function
        self.args = args
        self.type = _converted_types(type)


class Column(object):

    def __init__(self, col):
        self.column = col
        self.function = None
        self.index = None

    def cast(self, cast):
        from ddf_library.functions.etl.attributes_changer import \
            with_column_cast

        settings = dict()
        settings['attributes'] = [self.column]
        settings['cast'] = [cast]

        operation = [with_column_cast, settings.copy()]
        self.function = operation
        return self

    def _get_index(self, df):
        if self.index is None:
            self.index = df.columns.get_loc(self.column)
        return self.index

    def to_datetime(self, mask):
        from ddf_library.functions.etl.attributes_changer import \
            with_column_cast
        settings = dict()
        settings['attributes'] = [self.column]
        settings['cast'] = [DateType]
        settings['datetime_format'] = mask

        operation = [with_column_cast, settings.copy()]
        self.function = operation
        return self


col = Column
