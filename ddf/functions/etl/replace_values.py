#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__ = "Lucas Miguel S Ponce"
__email__ = "lucasmsp@gmail.com"


import numpy as np


def replace_value(data, settings):
    """
    Replace one or more values to new ones in a pandas's DataFrame.

    :param data: A list with nfrag pandas's dataframe;
    :param settings: A dictionary that contains:
    - replaces: A dictionary where each key is a column to perform
        an operation. Each key is linked to a matrix of 2xN.
        The first row is respect to the old values (or a regex)
        and the last is the new values.
    - regex: True, to use a regex expression, otherwise is False.
        Can be used only if the columns are strings (default, False);
    :return: Returns a list with nfrag pandas's dataframe
    """

    replaces = settings['replaces']
    subset = settings.get('subset', data.columns)

    to_replace = dict()
    for col in subset:
        to_replace[col] = replaces

    regexes = settings.get('regex', False)  # only if is string

    data.replace(to_replace=to_replace, regex=regexes, inplace=True)

    # if not regexes:
    # tmp_o = []
    # tmp_v = []
    # ixs = []
    # for ix in range(len(olds_v)):
    #     if isinstance(olds_v[ix], float):
    #         tmp_o.append(olds_v[ix])
    #         tmp_v.append(news_v[ix])
    #         ixs.append(ix)
    # olds_v = [olds_v[ix] for ix in range(len(olds_v))
    #           if ix not in ixs]
    # news_v = [news_v[ix] for ix in range(len(news_v))
    #           if ix not in ixs]

    # replace might not work with floats because the
    # floating point representation you see in the repr of
    # the DataFrame might not be the same as the underlying
    # float. Because of that, we need to perform float
    # operations in separate way.

    # for old, new in zip(tmp_o, tmp_v):
    #     mask = np.isclose(data[col],  old, rtol=1e-06)
    #     data.ix[mask, col] = new

    info = [data.columns.tolist(), data.dtypes.values, [len(data)]]

    return data, info


def preprocessing(settings):
    """Check all the settings."""
    replaces = settings.get('replaces', {})
    if not isinstance(replaces, dict):
        raise Exception('You must inform a valid replaces settings !')
    return settings

