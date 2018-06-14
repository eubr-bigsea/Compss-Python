#!/usr/bin/python
# -*- coding: utf-8 -*-
"""AttributesChanger: Rename or change the data's type of some columns."""

__author__ = "Lucas Miguel S Ponce"
__email__ = "lucasmsp@gmail.com"

from pycompss.api.task import task
import pandas as pd


class AttributesChangerOperation(object):
    """Attributes Changer Operation.

     Rename or change the data's type of some columns.
     """

    def transform(self, data, settings, nfrag):
        """AttributesChangerOperation.

        :param data: A list with numFrag pandas's dataframe;
        :param settings: A dictionary that contains:
            - attributes: A list of column(s) to be changed (or renamed).
            - new_name: A list of alias with the same size of attributes.
            If empty, the operation will overwrite all columns in attributes
            (default, empty);
            - new_data_type: The new type of the selected columns:
                * 'keep' - (default);
                * 'integer';
                * 'string';
                * 'double';
                * 'Date';
                * 'Date/time';
        :param nfrag: The number of fragments;
        :return: Returns a list with numFrag pandas's dataframe.
        """

        settings = self.preprocessing(settings)
        result = [[] for _ in range(nfrag)]
        for f in range(nfrag):
            result[f] = _change_attribute(data[f], settings)
        return result

    def preprocessing(self, settings):
        """Validate the parameters"""
        if 'attributes' not in settings:
            raise Exception("You must inform an `attributes` column.")

        attributes = settings['attributes']
        alias = settings.get('new_name', attributes)
        if len(alias) > 0 and (len(alias) != len(attributes)):
            raise Exception("Alias and attributes must have the same length, "
                            " or Alias must be a empty list.")
        return settings

    def transform_serial(self, data, settings):
        """Peform an attribute changer in each fragment."""
        return _change_attribute_(data, settings)


@task(returns=list)
def _change_attribute(data, settings):
    return _change_attribute_(data, settings)


def _change_attribute_(data, settings):
    """Peform an attribute changer in each fragment."""

    attributes = settings['attributes']
    new_name = settings.get('new_name', attributes)
    new_data_type = settings.get('new_data_type', 'keep')
    cols = data.columns
    for col in attributes:
        if col not in cols:
            raise Exception("The column `{}` dont exists!.".format(col))

    # first, change the data types.
    for att in attributes:
        if new_data_type == 'keep':
            pass
        elif new_data_type == 'integer':
            data[att] = data[att].astype(int)
        elif new_data_type == 'string':
            data[att] = data[att].astype(str)
        elif new_data_type == "double":
            data[att] = \
                pd.to_numeric(data[att], downcast='float', errors='coerce')
        elif new_data_type == "Date":
            tmp = pd.to_datetime(data[att])
            data[att] = tmp.apply(convertToDate)
        elif new_data_type == "Date/time":
            data[att] = pd.to_datetime(data[att], infer_datetime_format=True)

    # second, rename the columns
    mapper = dict(zip(attributes, new_name))
    data.rename(columns=mapper, inplace=True)
    return data


def convertToDate(col):
    """Convert datetime to date."""
    return col.date()
