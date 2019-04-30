#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Lucas Miguel S Ponce"
__email__ = "lucasmsp@gmail.com"

from ddf_library.utils import generate_info

import pandas as pd
import numpy as np


def explode(data, settings):
    """
    Split a column where the content is a list.

    :param data: pandas's DataFrame;
    :param settings: a dictionary with:
        - 'column': The column name which will be exploded;
    :return: A pandas's DataFrame.
    """
    column = settings['column']
    frag = settings['id_frag']

    if column in data.columns and len(data) > 0:
        data = unnest(data, column)

    info = generate_info(data, frag)
    return data, info


def unnest(df, col):
    # https://stackoverflow.com/questions/45885143/explode-lists-with
    # -different-lengths-in-pandas
    x = df.iloc[:, :-1].values.repeat(df[col].apply(len), 0)
    y = df[col].apply(pd.Series).stack().values.reshape(-1, 1)

    return pd.DataFrame(np.hstack((x, y)), columns=df.columns)
