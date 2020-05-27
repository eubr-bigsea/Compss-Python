#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from ddf_library.bases.metadata import OPTGroup


class Operation(object):

    def __init__(self, settings=None):
        if settings is None:
            settings = {}
        self.settings = settings

        self.start_columns = []
        self.current_columns = []
        self.end_columns = []

        self.start_fields = []
        self.current_fields = []
        self.end_fields = []


class NoOp(Operation):

    def __str__(self):
        return "Not mapped yet!"


class Select(Operation):

    phi_category = OPTGroup.OPT_SERIAL
    n_output = 1
    n_input = 1
    require_info = False

    def __init__(self, settings=None):
        super().__init__(settings)
        self.current_columns = self.settings['columns']

    @staticmethod
    def function(df, params):
        from ddf_library.functions.etl.select import select
        return select(df, params)

    def __str__(self):
        cols = ','.join(self.settings['columns'])
        in_fields = ["{}=${}".format(c, f) for c, f in zip(self.start_columns,
                                                           self.start_fields)]
        out_fields = ["{}=${}".format(c, f) for c, f in zip(self.end_columns,
                                                            self.end_fields)]
        return 'Select({})[inputCols:{} | OutputCols:{}]'.format(
                cols, ",".join(in_fields), ",".join(out_fields))


class DropColumns(Operation):

    phi_category = OPTGroup.OPT_SERIAL
    n_output = 1
    n_input = 1
    require_info = False

    def __init__(self, settings=None):
        super().__init__(settings)
        self.current_columns = self.settings['columns']

    @staticmethod
    def function(df, params):
        from ddf_library.functions.etl.drop import drop
        return drop(df, params)

    def __str__(self):
        cols = ','.join(self.settings['columns'])
        in_fields = ["{}=${}".format(c, f) for c, f in zip(self.start_columns,
                                                           self.start_fields)]
        out_fields = ["{}=${}".format(c, f) for c, f in zip(self.end_columns,
                                                            self.end_fields)]
        return 'DropColumns({})[inputCols:{} | OutputCols:{}]'.format(
                cols, ",".join(in_fields), ",".join(out_fields))


class Replace(Operation):

    phi_category = OPTGroup.OPT_SERIAL
    n_output = 1
    n_input = 1
    require_info = False

    def __init__(self, settings=None):
        super().__init__(settings)
        self.preprocessing()
        self.current_columns = self.settings['subset']

    @staticmethod
    def function(df, params):
        from ddf_library.functions.etl.replace_values import replace_value
        return replace_value(df, params)

    def preprocessing(self):
        from ddf_library.functions.etl.replace_values import preprocessing
        self.settings = preprocessing(self.settings)

    def __str__(self):
        col = ",".join(self.settings['subset'])
        return 'Replace({att})'.format(att=col)


class Filter(Operation):

    phi_category = OPTGroup.OPT_SERIAL
    n_output = 1
    n_input = 1
    require_info = False

    def __init__(self, settings=None):
        super().__init__(settings)
        query = self.settings['query']

        def get_terms(expr):
            import re
            terms = re.sub('[!()&|><=]', '', expr)
            terms = re.sub(r'\".+\"', '', terms)
            terms = re.sub(r"\'.+\'", '', terms).split(" ")
            terms = [t for t in terms if len(t) > 0 and not t.isdigit()]
            return terms

        self.current_columns = get_terms(query)

    @staticmethod
    def function(df, params):
        from ddf_library.functions.etl.filter import filter_rows
        return filter_rows(df, params)

    def __str__(self):
        in_fields = ["{}=${}".format(c, f) for c, f in zip(self.start_columns,
                                                           self.start_fields)]
        out_fields = ["{}=${}".format(c, f) for c, f in zip(self.end_columns,
                                                            self.end_fields)]
        return 'Filter({})[inputCols:{} | OutputCols:{}]'.format(
                ",".join(self.current_columns),
                ",".join(in_fields),
                ",".join(out_fields))


class DropNaRows(Operation):

    phi_category = OPTGroup.OPT_SERIAL
    n_output = 1
    n_input = 1
    require_info = False

    def __init__(self, settings=None):
        super().__init__(settings)
        self.current_columns = self.settings['attributes']

    @staticmethod
    def function(df, params):
        from ddf_library.functions.etl.clean_missing import drop_nan_rows
        return drop_nan_rows(df, params)

    def __str__(self):
        col = ",".join(self.settings['attributes'])
        in_fields = ["{}=${}".format(c, f) for c, f in zip(self.start_columns,
                                                           self.start_fields)]
        out_fields = ["{}=${}".format(c, f) for c, f in zip(self.end_columns,
                                                            self.end_fields)]

        return 'DropNaRows({})[inputCols:{} | OutputCols:{}]'.format(
                col, ",".join(in_fields), ",".join(out_fields))


class ReadFromFile(Operation):

    def __str__(self):
        return 'ReadFromFile({incol})[?]'.format(incol=''.join(self.output_col))


class Map(Operation):
    phi_category = OPTGroup.OPT_SERIAL
    n_output = 1
    n_input = 1
    require_info = False

    def __init__(self, settings=None):
        super().__init__(settings)
        self.current_columns = [self.settings['alias']]

    @staticmethod
    def function(df, params):
        from ddf_library.functions.etl.map import map
        return map(df, params)

    def __str__(self):
        in_fields = ["{}=${}".format(c, f) for c, f in zip(self.start_columns,
                                                           self.start_fields)]
        out_fields = ["{}=${}".format(c, f) for c, f in zip(self.end_columns,
                                                            self.end_fields)]
        return 'Map({})[inputCols:{} | OutputCols:{}]'.format(
                self.settings['alias'], ",".join(in_fields),
                ",".join(out_fields))


class Parallelize(Operation):

    phi_category = OPTGroup.OPT_OTHER
    n_output = 1
    n_input = 0
    require_info = False

    def __init__(self, settings=None):
        super().__init__(settings)
        self.current_columns = self.settings['input data'].columns.tolist()

    @staticmethod
    def function(_, params):
        from ddf_library.functions.etl.parallelize import parallelize
        df = params['input data']
        return parallelize(df, params)

    def __str__(self):
        in_fields = ["{}=${}".format(c, f) for c, f in zip(self.start_columns,
                                                           self.start_fields)]
        out_fields = ["{}=${}".format(c, f) for c, f in zip(self.end_columns,
                                                            self.end_fields)]
        return 'Parallelize(DataFrame)[inputCols:{} | OutputCols:{}]'\
            .format(",".join(in_fields), ",".join(out_fields))


class ImportCOMPSsData(Operation):

    phi_category = OPTGroup.OPT_OTHER
    n_output = 1
    n_input = 0
    require_info = False

    def __init__(self, settings=None):
        super().__init__(settings)
        self.current_columns = list(self.settings['input data'][0].columns)

    @staticmethod
    def function(_, params):
        from ddf_library.functions.etl.parallelize import import_to_ddf
        df = params['input_data']
        parquet = params['parquet']
        schema = params['schema']
        return import_to_ddf(df, parquet=parquet, schema=schema)

    def __str__(self):
        in_fields = ["{}=${}".format(c, f) for c, f in zip(self.start_columns,
                                                           self.start_fields)]
        out_fields = ["{}=${}".format(c, f) for c, f in zip(self.end_columns,
                                                            self.end_fields)]
        return 'ImportCOMPSsData(List of DataFrames)' \
               '[inputCols:{} | OutputCols:{}]'.format(
                ",".join(in_fields), ",".join(out_fields))
