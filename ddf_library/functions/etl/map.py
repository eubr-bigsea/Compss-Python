#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Lucas Miguel S Ponce"
__email__ = "lucasmsp@gmail.com"

import ddf_library.bases.config as config
from ddf_library.utils import generate_info
import numpy as np

from ddf_library.columns import Column, udf


def map(data, settings):
    """
    Returns a new DataFrame applying the expression to the specified column.

    :param data:  A pandas's DataFrame;
    :param settings: A dictionary that contains:
        - function: A lambda function;
        - alias: New column name;
    :return: Returns pandas's DataFrame with the news columns.

    .. seealso::
    https://engineering.upside.com/a-beginners-guide-to-optimizing-pandas-
    code-for-speed-c09ef2c6a4d6
    """
    config.columns = data.columns.tolist()  # todo: remove
    function = settings['function']
    new_column = settings['alias']
    frag = settings['id_frag']
    size = len(data)

    if size > 0:

        if isinstance(function, Column):
            # if is passed only col('oi')
            func, settings_intern = function.function
            settings_intern['alias'] = [new_column]
            settings_intern['id_frag'] = frag
            data, _ = func(data, settings_intern)

        elif isinstance(function, udf):
            func = function.function
            args = function.args
            dtype = function.type

            # index where arg is a Column object
            idx, mapper = list(), dict()
            for i, a in enumerate(args):
                if isinstance(a, Column):
                    idx.append(i)
                    mapper[a] = a._get_index(data)

            if isinstance(dtype, list):
                output = [0] * size
            else:
                output = np.zeros(size, dtype=dtype)

            for i, row in enumerate(data.to_numpy()):
                # row = [row[f._get_index(data)]
                #        if isinstance(f, Column) else f for f in args]

                current_args = [row[mapper[a]]
                                if i in idx else a
                                for i, a in enumerate(args)]
                output[i] = func(*current_args)

            data[new_column] = output
        else:
            raise Exception('You must inform a column or a udf operation.')

    else:
        data[new_column] = np.nan

    info = generate_info(data, frag)
    return data, info
