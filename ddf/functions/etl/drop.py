#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__ = "Lucas Miguel S Ponce"
__email__ = "lucasmsp@gmail.com"


def drop(data, columns):
    """
    Returns a new DataFrame that drops the specified column.
    Nothing is done if schema doesn't contain the given column name(s).

    :param data: A pandas's DataFrame;
    :param columns: A list with the columns names to be removed;
    :return: A pandas's DataFrame.
    """

    if len(columns) == 0:
        raise Exception("You should pass at least one query.")

    result = data.drop(columns, axis=1)
    info = [result.columns.tolist(), result.dtypes.values, [len(result)]]
    return result, info




