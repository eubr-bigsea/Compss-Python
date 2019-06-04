#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Lucas Miguel S Ponce"
__email__ = "lucasmsp@gmail.com"

from ddf_library.utils import generate_info


def select(data, settings):
    """
    Projects a set of expressions and returns a new DataFrame.

    :param data: pandas's DataFrame;
    :param settings:
        - 'columns': A list with the columns names which will be selected;
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
    del data

    info = generate_info(result, frag)
    return result, info


def select_exprs(data, settings):
    # noinspection PyUnresolvedReferences
    """
    Projects a set of expressions and returns a new DataFrame. This is a
    variant of select() that accepts expressions.


    :param data: pandas's DataFrame;
    :param settings:
     - 'exprs': SQL expressions. The column names are keywords;
    :return: A pandas's DataFrame with only the selected columns.

    .. note:: These operations are supported by select_exprs:

          * Arithmetic operations except for the left shift (<<) and
            right shift (>>) operators,
             e.g., 'col' + 2 * pi / s ** 4 % 42 - the_golden_ratio

          * list and tuple literals, e.g., [1, 2] or (1, 2)
          * Math functions: sin, cos, exp, log, abs, log10, ...
          * You must explicitly reference any local variable that you want to
            use in an expression by placing the @ character in front of the
            name.
          * This Python syntax is not allowed:

           - Function calls other than math functions.
           - is/is not operations
           - if expressions
           - lambda expressions
           - list/set/dict comprehensions
           - Literal dict and set expressions
           - yield expressions
           - Generator expressions
           - Boolean expressions consisting of only scalar values
           - Statements: Neither simple nor compound statements are allowed.

    .. seealso:: Visit this `link <https://pandas-docs.github.io/pandas-docs
       -travis/reference/api/pandas.eval.html#pandas.eval>`__ to more
       information about eval options.

    """

    frag = settings['id_frag']
    exprs = settings['exprs']

    cols_select = []
    for expr in exprs:
        if '=' in expr:
            data.eval(expr, inplace=True)
            cols_select.append(len(data.columns)-1)
        elif expr in data.columns:
            cols_select.append(data.columns.get_loc(expr))

    data = data.iloc[:, cols_select]

    info = generate_info(data, frag)
    return data, info
