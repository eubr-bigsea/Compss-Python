#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AttributesChanger: Rename or change the data's type of some columns."""

__author__ = "Lucas Miguel S Ponce"
__email__ = "lucasmsp@gmail.com"

from ddf_library.utils import generate_info
from pycompss.api.task import task
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
         'double', 'date', 'date/time';

    :return: A pandas's DataFrame.
    """

    attributes = settings['attributes']
    new_data_type = settings['cast']
    frag = settings['id_frag']

    cols = data.columns
    for col in attributes:
        if col not in cols:
            raise Exception("The column `{}` dont exists!.".format(col))

    # first, change the data types.
    for att, dtype in zip(attributes, new_data_type):
        if dtype == 'integer':
            data[att] = data[att].astype(int)
        elif dtype == 'string':
            data[att] = data[att].astype(str)
        elif dtype == "double":
            data[att] = \
                pd.to_numeric(data[att], downcast='float', errors='coerce')
        elif dtype == "date":
            tmp = pd.to_datetime(data[att])
            data[att] = tmp.apply(convert_to_date)
        elif dtype == "date/time":
            data[att] = pd.to_datetime(data[att], infer_datetime_format=True)

    info = generate_info(data, frag)

    return data, info


def convert_to_date(col):
    """Convert datetime to date."""
    return col.date()
