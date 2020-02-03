#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Lucas Miguel S Ponce"
__email__ = "lucasmsp@gmail.com"

import ddf_library.bases.config as config
from ddf_library.utils import generate_info
import numpy as np
import importlib


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
    config.columns = data.columns.tolist()
    function = settings['function']
    new_column = settings['alias']
    frag = settings['id_frag']

    size = len(data)
    if size > 0:
        # data[new_column] = data.apply(function, axis=1)
        output = [0] * size

        import ddf_library.utils
        importlib.reload(ddf_library.utils)

        for i, row in enumerate(data.values):
            output[i] = function(row)

        data[new_column] = output
    else:
        data[new_column] = np.nan

    info = generate_info(data, frag)
    return data, info
