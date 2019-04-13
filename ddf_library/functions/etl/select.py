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
    """
    Projects a set of expressions and returns a new DataFrame. This is a
    variant of select() that accepts expressions.


    :param data: pandas's DataFrame;
    :param settings:
     - 'exprs': SQL expressions. The column names are keywords;
    :return: A pandas's DataFrame with only the columns choosed.

    .. note: These operations are supported by select_exprs:

    * Arithmetic operations except for the left shift (<<) and right shift (>>)
     operators, e.g., df + 2 * pi / s ** 4 % 42 - the_golden_ratio
    * Comparison operations, including chained comparisons, e.g., 2 < df < df2
    * Boolean operations, e.g., df < df2 and df3 < df4 or not df_bool
    * list and tuple literals, e.g., [1, 2] or (1, 2)
    * Attribute access, e.g., df.a
    * Subscript expressions, e.g., df[0]
    * Simple variable evaluation, e.g., pd.eval('df') (this is not very useful)
    * Math functions: sin, cos, exp, log, expm1, log1p, sqrt, sinh, cosh, tanh,
     arcsin, arccos, arctan, arccosh, arcsinh, arctanh, abs, arctan2 and log10.

    * This Python syntax is not allowed:

     * Expressions

      - Function calls other than math functions.
      - is/is not operations
      - if expressions
      - lambda expressions
      - list/set/dict comprehensions
      - Literal dict and set expressions
      - yield expressions
      - Generator expressions
      - Boolean expressions consisting of only scalar values

     * Statements: Neither simple nor compound statements are allowed.
      This includes things like for, while, and if.


    You must explicitly reference any local variable that you want to use in an
    expression by placing the @ character in front of the name.

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
