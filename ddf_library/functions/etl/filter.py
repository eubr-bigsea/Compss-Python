#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Lucas Miguel S Ponce"
__email__ = "lucasmsp@gmail.com"

from ddf_library.utils import generate_info


def filter_rows(data, settings):
    """
    Filters rows using the given condition.

    :param data: A pandas's DataFrame;
    :param settings: a dictionary with:
     - query: A valid query.
    :return: A pandas's DataFrame.

    .. seealso:: Visit this `link <https://pandas.pydata.org/pandas-docs/
         stable/generated/pandas.DataFrame.query.html>`__ to more
         information about query options.
    """
    query = settings['query']
    frag = settings['id_frag']

    if len(query) == 0:
        raise Exception("You should pass at least one query.")

    result = data.query(query)
    info = generate_info(result, frag)
    return result, info
