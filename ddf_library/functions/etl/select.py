#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Lucas Miguel S Ponce"
__email__ = "lucasmsp@gmail.com"

from ddf_library.utils import generate_info


def select(data, settings):
    """
    Projects a set of expressions and returns a new DataFrame.

    :param data: pandas's DataFrame;
    :param columns: A list with the columns names which will be selected;
    :return: A pandas's DataFrame with only the columns choosed.
    """
    columns = settings['columns']
    frag = settings['id_frag']
    # remove the columns that not in list1
    fields = [field for field in columns if field in data.columns]
    if len(fields) == 0:
        raise Exception("The columns passed as parameters "
                        "do not belong to this dataframe.")

    result = data[fields]

    info = generate_info(result, frag)
    return result, info


