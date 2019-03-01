#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__ = "Lucas Miguel S Ponce"
__email__ = "lucasmsp@gmail.com"


def select(data, columns):
    """
    Projects a set of expressions and returns a new DataFrame.

    :param data: pandas's DataFrame;
    :param columns: A list with the columns names which will be selected;
    :return: A pandas's DataFrame with only the columns choosed.
    """

    # remove the columns that not in list1
    fields = [field for field in columns if field in data.columns]
    if len(fields) == 0:
        raise Exception("The columns passed as parameters "
                        "do not belong to this dataframe.")

    result = data[fields]

    info = [result.columns.tolist(), result.dtypes.values, [len(result)]]
    return result, info


