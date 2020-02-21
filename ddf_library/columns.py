#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Lucas Miguel S Ponce"
__email__ = "lucasmsp@gmail.com"


__all__ = ['col', 'udf']


from ddf_library.types import DateType, _converted_types
import pandas as pd


class Udf(object):

    def __init__(self, function, dtype, *args):
        self.function = function
        self.args = args
        self.type = _converted_types[dtype]


class Column(object):

    def __init__(self, column):
        self.column = column
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

    # ---------------------- DateTime operations ------------------------------

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

    def year(self):
        """
        Extract the year of a given date as integer.
        :return: integer
        """
        def get_year(df, params):
            att = params['attributes']
            alias = params['alias'][0]
            df[alias] = pd.DatetimeIndex(df[att]).year
            return df, {}

        settings = {'attributes': self.column}
        self.function = [get_year, settings]
        return self

    def mouth(self):
        """
        Extract the month of a given date as integer.
        :return: integer
        """
        def get_mouth(df, params):
            att = params['attributes']
            alias = params['alias'][0]
            df[alias] = pd.DatetimeIndex(df[att]).mouth
            return df, {}

        settings = {'attributes': self.column}
        self.function = [get_mouth, settings]
        return self

    def day(self):
        """
        Extract the day of a given date as integer.
        :return: integer
        """
        def get_day(df, params):
            att = params['attributes']
            alias = params['alias'][0]
            df[alias] = pd.DatetimeIndex(df[att]).day
            return df, {}

        settings = {'attributes': self.column}
        self.function = [get_day, settings]
        return self

    def hour(self):
        """
        Extract the hours of a given date as integer.
        :return: integer
        """

        def get_hour(df, params):
            att = params['attributes']
            alias = params['alias'][0]
            df[alias] = pd.DatetimeIndex(df[att]).hour
            return df, {}

        settings = {'attributes': self.column}
        self.function = [get_hour, settings]
        return self

    def minute(self):
        """
        Extract the minutes of a given date as integer.
        :return: integer
        """

        def get_minute(df, params):
            att = params['attributes']
            alias = params['alias'][0]
            df[alias] = pd.DatetimeIndex(df[att]).minute
            return df, {}

        settings = {'attributes': self.column}
        self.function = [get_minute, settings]
        return self

    def second(self):
        """
        Extract the seconds of a given date as integer.
        :return: integer
        """

        def get_second(df, params):
            att = params['attributes']
            alias = params['alias'][0]
            df[alias] = pd.DatetimeIndex(df[att]).second
            return df, {}

        settings = {'attributes': self.column}
        self.function = [get_second, settings]
        return self

    def date(self):
        """
        Extract the date of a given date.
        :return: date
        """

        def get_date(df, params):
            att = params['attributes']
            alias = params['alias'][0]
            df[alias] = pd.DatetimeIndex(df[att]).date
            return df, {}

        settings = {'attributes': self.column}
        self.function = [get_date, settings]
        return self

    def time(self):
        """
        Extract the time of a given date.
        :return: time
        """

        def get_time(df, params):
            att = params['attributes']
            alias = params['alias'][0]
            df[alias] = pd.DatetimeIndex(df[att]).time
            return df, {}

        settings = {'attributes': self.column}
        self.function = [get_time, settings]
        return self

    def dayofyear(self):
        """
        Extract the day of the year of a given date as integer.
        :return: integer
        """

        def get_dayofyyear(df, params):
            att = params['attributes']
            alias = params['alias'][0]
            df[alias] = pd.DatetimeIndex(df[att]).dayofyear
            return df, {}

        settings = {'attributes': self.column}
        self.function = [get_dayofyyear, settings]
        return self

    def weekofyear(self):
        """
        Extract the week ordinal of the year of a given date as integer.
        :return: integer
        """

        def get_weekofyear(df, params):
            att = params['attributes']
            alias = params['alias'][0]
            df[alias] = pd.DatetimeIndex(df[att]).weekofyear
            return df, {}

        settings = {'attributes': self.column}
        self.function = [get_weekofyear, settings]
        return self

    def week(self):
        """
        Extract the week of a given date as integer.
        :return: integer
        """

        def get_week(df, params):
            att = params['attributes']
            alias = params['alias'][0]
            df[alias] = pd.DatetimeIndex(df[att]).week
            return df, {}

        settings = {'attributes': self.column}
        self.function = [get_week, settings]
        return self

    def dayofweek(self):
        """
        Extract the day of the week with Monday=0 and Sunday=6 as integer.
        :return: integer
        """

        def get_dayofweek(df, params):
            att = params['attributes']
            alias = params['alias'][0]
            df[alias] = pd.DatetimeIndex(df[att]).dayofweek
            return df, {}

        settings = {'attributes': self.column}
        self.function = [get_dayofweek, settings]
        return self


col = Column
udf = Udf
