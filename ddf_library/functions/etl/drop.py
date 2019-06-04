#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Lucas Miguel S Ponce"
__email__ = "lucasmsp@gmail.com"

from ddf_library.utils import generate_info


def drop(data, settings):
    """
    Returns a new DataFrame that drops the specified column.
    Nothing is done if schema doesn't contain the given column name(s).

    :param data: A pandas's DataFrame;
    :param settings: A dictionary with:
     - columns: A list with the columns names to be removed;
    :return: A pandas's DataFrame.
    """

    columns, frag = settings['columns'], settings['id_frag']

    if len(columns) == 0:
        raise Exception("You should pass at least one query.")

    data.drop(columns, axis=1, inplace=True)
    info = generate_info(data, frag)
    return data, info
