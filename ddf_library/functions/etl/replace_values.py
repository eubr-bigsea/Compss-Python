#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Lucas Miguel S Ponce"
__email__ = "lucasmsp@gmail.com"

from ddf_library.utils import generate_info
import numpy as np


def replace_value(data, settings):
    """
    Replace one or more values to new ones in a pandas's DataFrame.

    :param data: A list of pandas's DataFrame;
    :param settings: A dictionary that contains:
        - replaces: A dictionary where each key is a column to perform
            an operation. Each key is linked to a matrix of 2xN.
            The first row is respect to the old values (or a regex)
            and the last is the new values.
        - regex: True, to use a regex expression, otherwise is False.
            Can be used only if the columns are strings (default, False);
    :return: Returns a list of pandas's DataFrame
    """

    replaces = settings['replaces']
    frag = settings['id_frag']
    subset = settings.get('subset', data.columns)

    to_replace = dict()
    for col in subset:
        to_replace[col] = replaces

    regex = settings.get('regex', False)  # only if is string

    data.replace(to_replace=to_replace, regex=regex, inplace=True)

    info = generate_info(data, frag)

    return data, info


def preprocessing(settings):
    """Check all the settings."""
    replaces = settings.get('replaces', {})
    if not isinstance(replaces, dict):
        raise Exception('You must inform a valid replaces settings !')
    return settings
