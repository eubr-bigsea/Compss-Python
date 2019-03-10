#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__ = "Lucas Miguel S Ponce"
__email__ = "lucasmsp@gmail.com"

import context
from .ddf import DDF


class GroupedDDF(DDF):

    """
    A set of methods for aggregations on a DDF, created by DDF.group_by().

    The available aggregate functions are:

    - avg: Computes average values for each numeric columns for each group;
    - mean: Alias for avg;
    - count: Counts the number of records for each group;
    - first': Returns the first element of group;
    - last': Returns the last element of group;
    - max': Computes the max value for each numeric columns for each group;
    - min': Computes the min value for each numeric column for each group;
    - sum': Computes the sum for each numeric columns for each group;
    - list': Returns a list of objects with duplicates;
    - set': Returns a set of objects with duplicate elements
    """

    def __init__(self, ddf_var):
        self.last_uuid = ddf_var.last_uuid
        self.ddf_var = ddf_var
        self.parameters = context.COMPSsContext() \
            .get_task_function(self.last_uuid)[1]

        super(GroupedDDF, self).__init__(task_list=ddf_var.task_list,
                                         last_uuid=self.last_uuid)

    def agg(self, exprs):
        """
        Compute aggregates and returns the result as a DDF.

        :param exprs: A single dict mapping from string to string, where the
         key is the column to perform aggregation on, and the value is a list
         of aggregation functions.

        :Example:

        >>> ddf1.group_by(group_by=['col_1']).agg({'col_2': ['sum', 'last']})
        """
        self.parameters['operation'] = exprs
        context.COMPSsContext.tasks_map[self.last_uuid]['function'][1] = \
            self.parameters

        return self.ddf_var

    def avg(self, cols=None):
        """
        Computes average values for each numeric columns for each group.

        :param cols: A list of columns

        :Example:

        >>> ddf1.group_by(group_by=['col_1']).avg(['col_2'])
        """
        self._apply_agg(cols, 'mean')
        return self

    def count(self, cols=None):
        """
        Counts the number of records for each group.

        :param cols: A list of columns

        :Example:

        >>> ddf1.group_by(group_by=['col_1']).count(['col_2'])
        """
        self._apply_agg(cols, 'count')
        return self

    def first(self, cols=None):
        """
        Returns the first element of group.

        :param cols: A list of columns

        :Example:

        >>> ddf1.group_by(group_by=['col_1']).avg(['col_2'])
        """
        self._apply_agg(cols, 'first')
        return self

    def last(self, cols=None):
        """
        Returns the last element of group.

        :param cols: A list of columns

        :Example:

        >>> ddf1.group_by(group_by=['col_1']).avg(['col_2'])
        """
        self._apply_agg(cols, 'last')
        return self

    def list(self, cols=None):
        """
        Returns a list of objects with duplicates;

        :param cols: A list of columns

        :Example:

        >>> ddf1.group_by(group_by=['col_1']).avg(['col_2'])
        """
        self._apply_agg(cols, 'list')
        return self

    def max(self, cols=None):
        """
        Computes the max value for each numeric columns for each group.

        :param cols: A list of columns

        :Example:

        >>> ddf1.group_by(group_by=['col_1']).avg(['col_2'])
        """
        self._apply_agg(cols, 'max')
        return self

    def mean(self, cols=None):
        """
        Alias for avg. Computes average values for each numeric columns for
        each group.

        :param cols: A list of columns

        :Example:

        >>> ddf1.group_by(group_by=['col_1']).mean(['col_2'])
        """
        self._apply_agg(cols, 'mean')
        return self

    def min(self, cols=None):
        """
        Computes the min value for each numeric column for each group.

        :param cols: A list of columns

        :Example:

        >>> ddf1.group_by(group_by=['col_1']).min(['col_2'])
        """
        self._apply_agg(cols, 'min')
        return self

    def set(self, cols=None):
        """
        Returns a set of objects with duplicate elements.

        :param cols: A list of columns

        :Example:

        >>> ddf1.group_by(group_by=['col_1']).set(['col_2'])
        """
        self._apply_agg(cols, 'set')
        return self

    def sum(self, cols=None):
        """
        Computes the sum for each numeric columns for each group.

        :param cols: A list of columns

        :Example:

        >>> ddf1.group_by(group_by=['col_1']).sum(['col_2'])
        """
        self._apply_agg(cols, 'sum')
        return self

    def _apply_agg(self, cols, func):
        exprs = self.parameters['operation']
        for col in cols:
            if col not in exprs:
                exprs[col] = []
            exprs[col].append(func)
        self.parameters['operation'] = exprs

        context.COMPSsContext.tasks_map[self.last_uuid]['function'][1] = \
            self.parameters

