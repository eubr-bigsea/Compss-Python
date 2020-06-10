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

        self.tag = self.__class__.__name__


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


class CRSTTransform(Operation):

    phi_category = OPTGroup.OPT_SERIAL
    n_output = 1
    n_input = 1
    require_info = False

    def __init__(self, settings=None):
        super().__init__(settings)
        self.current_columns = [self.settings['lon_col'],
                                self.settings['lat_col']]

    @staticmethod
    def function(df, params):
        from ddf_library.functions.geo import crst_transform
        return crst_transform(df, params)


class WithColumn(Operation):

    phi_category = OPTGroup.OPT_SERIAL
    n_output = 1
    n_input = 1
    require_info = False

    def __init__(self, settings=None):
        super().__init__(settings)
        from ddf_library.functions.etl.attributes_changer import \
            create_settings_cast
        column, cast = settings['column'], settings['cast']
        self.settings = create_settings_cast(attributes=column, cast=cast)
        self.current_columns = column

    @staticmethod
    def function(df, params):
        from ddf_library.functions.etl.attributes_changer import \
            with_column_cast
        return with_column_cast(df, params)


class FillNaByValue(Operation):

    phi_category = OPTGroup.OPT_SERIAL
    n_output = 1
    n_input = 1
    require_info = False

    def __init__(self, settings=None):
        super().__init__(settings)
        self.current_columns = settings['attributes']

    @staticmethod
    def function(df, params):
        from ddf_library.functions.etl.clean_missing import fill_by_value
        return fill_by_value(df, params)


class Explode(Operation):

    phi_category = OPTGroup.OPT_SERIAL
    n_output = 1
    n_input = 1
    require_info = False

    def __init__(self, settings=None):
        super().__init__(settings)
        self.current_columns = settings['column']

    @staticmethod
    def function(df, params):
        from ddf_library.functions.etl.explode import explode
        return explode(df, params)


class WithColumnRenamed(Operation):

    phi_category = OPTGroup.OPT_SERIAL
    n_output = 1
    n_input = 1
    require_info = False

    def __init__(self, settings=None):
        old_column, new_column = settings['old_column'], settings['new_column']

        if not isinstance(old_column, list):
            old_column = [old_column]

        if not isinstance(new_column, list):
            new_column = [new_column]
        settings = {'old_column': old_column, 'new_column': new_column}

        super().__init__(settings)
        self.current_columns = old_column

    @staticmethod
    def function(df, params):
        from ddf_library.functions.etl.attributes_changer import \
            with_column_renamed
        return with_column_renamed(df, params)


class SelectExprs(Operation):

    phi_category = OPTGroup.OPT_SERIAL
    n_output = 1
    n_input = 1
    require_info = False

    def __init__(self, settings=None):
        super().__init__(settings)
        self.current_columns = []

    @staticmethod
    def function(df, params):
        from ddf_library.functions.etl.select import select_exprs
        return select_exprs(df, params)


class WorkloadBalancer(Operation):

    phi_category = OPTGroup.OPT_OTHER
    n_output = 1
    n_input = 1
    require_info = True

    def __init__(self, settings=None):
        super().__init__(settings)
        self.current_columns = []

    @staticmethod
    def function(df, params):
        from ddf_library.functions.etl.balancer import WorkloadBalancer
        return WorkloadBalancer(params).transform(df)


class AddColumn(Operation):

    phi_category = OPTGroup.OPT_OTHER
    n_output = 1
    n_input = 1
    require_info = True

    def __init__(self, settings=None):
        super().__init__(settings)
        self.current_columns = []

    @staticmethod
    def function(df, params):
        from ddf_library.functions.etl.add_columns import AddColumnsOperation
        return AddColumnsOperation().transform(df[0], df[1], params)


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

    phi_category = OPTGroup.OPT_SERIAL
    n_output = 1
    n_input = 0
    require_info = False

    def __init__(self, settings=None, tag=""):
        super().__init__(settings)
        self.current_columns = [self.settings['alias']]

    def __str__(self):
        return 'ReadFromFile({incol})[?]'.format(incol=''.join(self.output_col))

    @staticmethod
    def function(df, params):
        pass


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


class Union(Operation):

    phi_category = OPTGroup.OPT_OTHER
    n_output = 1
    n_input = 2
    require_info = True

    def __init__(self, settings=None):
        super().__init__(settings)
        self.current_columns = []

    @staticmethod
    def function(df, params):
        from ddf_library.functions.etl.union import union
        return union(df[0], df[1], params)


class Repartition(Operation):

    phi_category = OPTGroup.OPT_OTHER
    n_output = 1
    n_input = 1
    require_info = True

    def __init__(self, settings=None):
        super().__init__(settings)
        self.current_columns = []

    @staticmethod
    def function(df, params):
        from ddf_library.functions.etl.repartition import repartition
        return repartition(df, params)


class RangePartition(Operation):

    phi_category = OPTGroup.OPT_OTHER
    n_output = 1
    n_input = 1
    require_info = True

    def __init__(self, settings=None):
        super().__init__(settings)
        self.current_columns = []

    @staticmethod
    def function(df, params):
        from ddf_library.functions.etl.range_partitioner import range_partition
        return range_partition(df, params)


class HashPartition(Operation):

    phi_category = OPTGroup.OPT_OTHER
    n_output = 1
    n_input = 1
    require_info = True

    def __init__(self, settings=None):
        super().__init__(settings)
        self.current_columns = []

    @staticmethod
    def function(df, params):
        from ddf_library.functions.etl.hash_partitioner import hash_partition
        return hash_partition(df, params)


class CrossTab(Operation):

    phi_category = OPTGroup.OPT_OTHER
    n_output = 1
    n_input = 1
    require_info = False

    def __init__(self, settings=None):
        super().__init__(settings)
        self.current_columns = []

    @staticmethod
    def function(df, params):
        from ddf_library.functions.statistics.cross_tab import cross_tab
        return cross_tab(df, params)


class CrossJoin(Operation):

    phi_category = OPTGroup.OPT_OTHER
    n_output = 1
    n_input = 2
    require_info = False

    def __init__(self, settings=None):
        super().__init__(settings)
        self.current_columns = []

    @staticmethod
    def function(df, _):
        from ddf_library.functions.etl.cross_join import cross_join
        return cross_join(df[0], df[1])


class Aggregation(Operation):

    phi_category = OPTGroup.OPT_LAST
    n_output = 1
    n_input = 1
    require_info = True

    def __init__(self, settings=None):
        super().__init__(settings)
        self.current_columns = []

    @staticmethod
    def function_first(df, params):
        from ddf_library.functions.etl.aggregation import aggregation_stage_1
        return aggregation_stage_1(df, params)

    @staticmethod
    def function_second(df, params):
        from ddf_library.functions.etl.aggregation import aggregation_stage_2
        return aggregation_stage_2(df, params)


class FillNan(Operation):

    phi_category = OPTGroup.OPT_LAST
    n_output = 1
    n_input = 1
    require_info = True

    def __init__(self, settings=None):
        super().__init__(settings)
        self.current_columns = []

    @staticmethod
    def function_first(df, params):
        from ddf_library.functions.etl.clean_missing import fill_nan_stage_1
        return fill_nan_stage_1(df, params)

    @staticmethod
    def function_second(df, params):
        from ddf_library.functions.etl.clean_missing import fill_nan_stage_2
        return fill_nan_stage_2(df, params)


class Subtract(Operation):

    phi_category = OPTGroup.OPT_LAST
    n_output = 1
    n_input = 2
    require_info = True

    def __init__(self, settings=None):
        super().__init__(settings)
        self.current_columns = []

    @staticmethod
    def function_first(df, params):
        from ddf_library.functions.etl.subtract import subtract_stage_1
        return subtract_stage_1(df[0], df[1], params)

    @staticmethod
    def function_second(df, params):
        from ddf_library.functions.etl.subtract import subtract_stage_2
        return subtract_stage_2(df[0], df[1], params)


class ExceptAll(Operation):

    phi_category = OPTGroup.OPT_LAST
    n_output = 1
    n_input = 2
    require_info = True

    def __init__(self, settings=None):
        super().__init__(settings)
        self.current_columns = []

    @staticmethod
    def function_first(df, params):
        from ddf_library.functions.etl.except_all import except_all_stage_1
        return except_all_stage_1(df[0], df[1], params)

    @staticmethod
    def function_second(df, params):
        from ddf_library.functions.etl.except_all import except_all_stage_2
        return except_all_stage_2(df[0], df[1], params)


class Distinct(Operation):

    phi_category = OPTGroup.OPT_LAST
    n_output = 1
    n_input = 1
    require_info = True

    def __init__(self, settings=None):
        super().__init__(settings)
        self.current_columns = []

    @staticmethod
    def function_first(df, params):
        from ddf_library.functions.etl.distinct import distinct_stage_1
        return distinct_stage_1(df, params)

    @staticmethod
    def function_second(df, params):
        from ddf_library.functions.etl.distinct import distinct_stage_2
        return distinct_stage_2(df, params)


class DropNaColumns(Operation):

    phi_category = OPTGroup.OPT_LAST
    n_output = 1
    n_input = 1
    require_info = True

    def __init__(self, settings=None):
        super().__init__(settings)
        self.current_columns = []

    @staticmethod
    def function_first(df, params):
        from ddf_library.functions.etl.clean_missing import \
            drop_nan_columns_stage_1
        return drop_nan_columns_stage_1(df, params)

    @staticmethod
    def function_second(df, params):
        from ddf_library.functions.etl.clean_missing import \
            drop_nan_columns_stage_2
        return drop_nan_columns_stage_2(df, params)


class GeoWithin(Operation):

    phi_category = OPTGroup.OPT_LAST
    n_output = 1
    n_input = 2
    require_info = True

    def __init__(self, settings=None):
        super().__init__(settings)
        self.current_columns = []

    @staticmethod
    def function_first(df, params):
        from ddf_library.functions.geo import geo_within_stage_1
        return geo_within_stage_1(df[0], df[1], params)

    @staticmethod
    def function_second(df, params):
        from ddf_library.functions.geo import geo_within_stage_2
        return geo_within_stage_2(df, params)


class Intersect(Operation):

    phi_category = OPTGroup.OPT_LAST
    n_output = 1
    n_input = 2
    require_info = True

    def __init__(self, settings=None):
        super().__init__(settings)
        self.current_columns = []

    @staticmethod
    def function_first(df, params):
        from ddf_library.functions.etl.intersect import intersect_stage_1
        return intersect_stage_1(df[0], df[1], params)

    @staticmethod
    def function_second(df, params):
        from ddf_library.functions.etl.intersect import intersect_stage_2
        return intersect_stage_2(df[0], df[1], params)


class Join(Operation):

    phi_category = OPTGroup.OPT_LAST
    n_output = 1
    n_input = 2
    require_info = True

    def __init__(self, settings=None):
        super().__init__(settings)
        self.current_columns = []

    @staticmethod
    def function_first(df, params):
        from ddf_library.functions.etl.join import join_stage_1
        return join_stage_1(df[0], df[1], params)

    @staticmethod
    def function_second(df, params):
        from ddf_library.functions.etl.join import join_stage_2
        return join_stage_2(df[0], df[1], params)


class Sample(Operation):

    phi_category = OPTGroup.OPT_LAST
    n_output = 1
    n_input = 2
    require_info = True

    def __init__(self, settings=None):
        super().__init__(settings)
        self.current_columns = []

    @staticmethod
    def function_first(df, params):
        from ddf_library.functions.etl.sample import sample_stage_1
        return sample_stage_1(df, params)

    @staticmethod
    def function_second(df, params):
        from ddf_library.functions.etl.sample import sample_stage_2
        return sample_stage_2(df, params)


class Sort(Operation):

    phi_category = OPTGroup.OPT_LAST
    n_output = 1
    n_input = 1
    require_info = True

    def __init__(self, settings=None):
        super().__init__(settings)
        self.current_columns = []

    @staticmethod
    def function_first(df, params):
        from ddf_library.functions.etl.sort import sort_stage_1
        return sort_stage_1(df, params)

    @staticmethod
    def function_second(df, params):
        from ddf_library.functions.etl.sort import sort_stage_2
        return sort_stage_2(df, params)


class Take(Operation):

    phi_category = OPTGroup.OPT_LAST
    n_output = 1
    n_input = 1
    require_info = True

    def __init__(self, settings=None):
        super().__init__(settings)
        self.current_columns = []

    @staticmethod
    def function_first(df, params):
        from ddf_library.functions.etl.take import take_stage_1
        return take_stage_1(df, params)

    @staticmethod
    def function_second(df, params):
        from ddf_library.functions.etl.take import take_stage_2
        return take_stage_2(df, params)


class RandomSplit(Operation):

    phi_category = OPTGroup.OPT_OTHER
    n_output = 2
    n_input = 1
    require_info = True

    def __init__(self, settings=None):
        super().__init__(settings)
        self.current_columns = []

    @staticmethod
    def function(df, params):
        from ddf_library.functions.etl.split import random_split
        return random_split(df, params)


class DataWriter(Operation):

    phi_category = OPTGroup.OPT_SERIAL
    n_output = 0
    n_input = 1
    require_info = False  #?:?

    def __init__(self, data_saver, settings, tag):
        super().__init__(settings)
        self.data_saver = data_saver
        self.tag = tag
        self.current_columns = []

    def function(self, df, params):
        return self.data_saver.save(df, params)


class DataReaderOperation(Operation):

    n_output = 1
    n_input = 0
    require_info = False

    def __init__(self, data_reader, settings, tag, opt=OPTGroup.OPT_SERIAL):
        super().__init__(settings)
        self.data_reader = data_reader
        self.tag = tag
        self.current_columns = []
        self.phi_category = opt
        if tag in ['read-many-file', 'read-hdfs']:
            self.blocks = data_reader.get_blocks()

    def function(self, _, params):

        if self.tag == 'read-many-file':
            return self.data_reader.transform_fs_distributed(self.blocks,
                                                             params)
        elif self.tag == 'read-one-file':
            return self.data_reader.transform_fs_single()
        else:
            return self.data_reader.transform_hdfs(self.blocks, params)


class ReadShapefile(Operation):

    n_output = 1
    n_input = 0
    require_info = False

    def __init__(self,  settings, opt=OPTGroup.OPT_LAST):
        super().__init__(settings)
        self.current_columns = []
        self.phi_category = opt

    @staticmethod
    def function_first(_, params):
        from ddf_library.functions.geo import read_shapefile_stage_1
        return read_shapefile_stage_1(params)

    @staticmethod
    def function_second(df, params):
        from ddf_library.functions.geo import read_shapefile_stage_2
        return read_shapefile_stage_2(df, params)


