#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__ = "Lucas Miguel S Ponce"
__email__ = "lucasmsp@gmail.com"

from pycompss.api.task import task


class SelectOperation(object):

    def transform(self, data, columns):
        """
        Projects a set of expressions and returns a new DataFrame.

        :param data: A list with nfrag pandas's dataframe;
        :param columns: A list with the columns names which will be selected;
        :return: A list with nfrag pandas's dataframe with only the
            columns choosed.
        """
        nfrag = len(data)
        result = [[] for _ in range(nfrag)]
        columns = self.preprocessing(columns)
        for f in range(nfrag):
            result[f] = _select(data[f], columns)

        return result

    def preprocessing(self, columns):
        if len(columns) == 0:
            raise Exception("You should pass at least one column.")
        return columns

    def transform_serial(self, list1, fields):
        """Perform a partial projection."""
        # remove the columns that not in list1
        fields = [field for field in fields if field in list1.columns]
        if len(fields) == 0:
            raise Exception("The columns passed as parameters "
                            "do not belong to this dataframe.")
        return list1[fields]


@task(returns=list)
def _select(list1, fields):
    """Perform a partial projection."""
    # remove the columns that not in list1
    fields = [field for field in fields if field in list1.columns]
    if len(fields) == 0:
        raise Exception("The columns passed as parameters "
                        "do not belong to this dataframe.")
    return list1[fields]


