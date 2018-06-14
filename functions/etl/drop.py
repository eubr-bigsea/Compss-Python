#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__ = "Lucas Miguel S Ponce"
__email__ = "lucasmsp@gmail.com"

from pycompss.api.task import task


class DropOperation(object):
    """DropOperation.

    Returns a new DataFrame that drops the specified column.
    Nothing is done if schema doesn't contain the given column name(s).
    """

    def transform(self, data, columns, nfrag):
        """transform.

        :param data: A list with nfrag pandas's dataframe;
        :param columns: A list with the columns names to be removed;
        :param nfrag: A number of fragments;
        :return: A list with nfrag pandas's dataframe.
        """
        result = [[] for _ in range(nfrag)]
        columns = self.preprocessing(columns)
        for f in range(nfrag):
            result[f] = _drop(data[f], columns)

        return result

    def preprocessing(self, columns):
        if len(columns) == 0:
            raise Exception("You should pass at least one query.")
        return columns

    def transform_serial(self, df, columns):
        """Peform a partial drop operation."""
        return df.drop(columns, axis=1)


@task(returns=list)
def _drop(df, columns):
    """Peform a partial drop operation."""
    return df.drop(columns, axis=1)


