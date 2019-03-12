#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__ = "Lucas Miguel S Ponce"
__email__ = "lucasmsp@gmail.com"


def filter(data, query):
    """
    Filters rows using the given condition.

    :param data: A pandas's DataFrame;
    :param query: A valid query.
    :return: A pandas's DataFrame.

    .. seealso:: Visit this `link <https://pandas.pydata.org/pandas-docs/
         stable/generated/pandas.DataFrame.query.html>`__ to more
         information about query options.
    """

    if len(query) == 0:
        raise Exception("You should pass at least one query.")

    result = data.query(query)
    info = [result.columns.tolist(), result.dtypes.values, [len(result)]]
    return result, info



