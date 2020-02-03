#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AttributesChanger: Rename or change the data's type of some columns."""

__author__ = "Lucas Miguel S Ponce"
__email__ = "lucasmsp@gmail.com"

from ddf_library.utils import generate_info
import pandas as pd


def with_column_renamed(data, settings):
    """
    Returns a new DataFrame by renaming an existing column.
    Nothing is done if schema doesn't contain the given column name(s).

    :param data: A pandas's DataFrame;
    :param settings: A dictionary that contains:

        * old_column: String or list of strings with columns to rename;
        * new_column: String or list of strings with new names.

    :return: A pandas's DataFrame.
    """

    existing = settings['old_column']
    new = settings['new_column']
    frag = settings['id_frag']

    mapper = dict(zip(existing, new))
    data.rename(columns=mapper, inplace=True)

    info = generate_info(data, frag)
    return data, info


def with_column_cast(data, settings):
    """
    Rename or change the data's type of some columns.

    :param data: A pandas's DataFrame;
    :param settings: A dictionary that contains:
        - attributes: A list of column(s) to cast;
        - cast: A list of strings with the supported types: 'integer', 'string',
         'decimal', 'date', 'date/time';

    :return: A pandas's DataFrame.
    """

    attributes = settings['attributes']
    new_data_type = settings['cast']
    frag = settings['id_frag']
    alias = settings.get('alias', None)

    if not isinstance(attributes, list):
        attributes = [attributes]

    if not isinstance(new_data_type, list):
        new_data_type = [new_data_type for _ in range(len(attributes))]

    if alias is None:
        alias = attributes

    if len(alias) != len(attributes):
        raise Exception('Alias and Attributes must have same length.')

    # TODO: optimize
    diff = len(new_data_type) - len(attributes)
    if diff > 0:
        cast = new_data_type[:len(attributes)]
    elif diff < 0:
        cast = new_data_type + ['keep' for _ in range(diff + 1)]

    cols = data.columns
    for col in attributes:
        if col not in cols:
            raise Exception("The column `{}` don't exists!.".format(col))

    # first, change the data types.
    for att, new_col, dtype in zip(attributes, alias, new_data_type):
        dtype = dtype.lower()
        if dtype == 'integer':
            data[new_col] = data[att].astype(int)
        elif dtype == 'string':
            data[new_col] = data[att].astype(str)
        elif dtype == "decimal":
            data[new_col] = \
                pd.to_numeric(data[att], downcast='float', errors='coerce')
        elif dtype == "date":
            data[new_col] = pd.to_datetime(data[att],
                                           infer_datetime_format=True).dt.date
        elif dtype == "date/time":
            data[new_col] = pd.to_datetime(data[att],
                                           infer_datetime_format=True)

    info = generate_info(data, frag)

    return data, info


def convert_to_date(col):
    """Convert datetime to date."""
    return col.date()
