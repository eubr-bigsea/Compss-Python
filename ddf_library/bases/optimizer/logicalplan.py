#!/usr/bin/env python3
# -*- coding: utf-8 -*-


__author__ = "Lucas Miguel S Ponce"
__email__ = "lucasmsp@gmail.com"

import ddf_library.bases.optimizer.operations as operations_list


class LogicalPlan(object):

    def __init__(self, catalog_tasks, operations, lineage):

        self.catalog_tasks = catalog_tasks
        self.lineage = lineage
        self.operations = operations

        # creates an id for each column
        self.columns_fields = []
        # maps all fields dependencies of a specific field
        self.field_dependecies = {}

        self.infer_columns()
        self.explain_logical_plan()

    def get_new_lineage(self):
        return self.lineage

    def convert_columns2fields(self, end_columns):
        fields = []
        for col in end_columns:
            self.columns_fields.append(col)
            fields.append(len(self.columns_fields)-1)
        return fields

    def infer_columns(self):

        # creates an id for each column
        self.columns_fields = []
        # maps all fields dependencies of a specific field
        self.field_dependecies = {}

        for uuid in self.lineage:
            n_input = self.operations[uuid].n_input
            if n_input == 0:
                end_columns = self.operations[uuid].current_columns
                self.operations[uuid].end_columns = end_columns
                end_fields = self.convert_columns2fields(end_columns)
                self.operations[uuid].end_fields = end_fields
            else:
                end_columns = self.operations[uuid].end_columns
                end_fields = self.operations[uuid].end_fields

            children = self.catalog_tasks.get_task_children(uuid)
            for child in children:
                self.operations[child].start_columns = end_columns
                self.operations[child].start_fields = end_fields
                op_name = self.operations[child].__class__.__name__

                if op_name == 'Map':
                    self.infer_to_map(child, end_columns, end_fields)

                elif op_name == 'DropColumns':
                    self.infer_to_dropcolumns(child, end_columns, end_fields)

                elif op_name == "Select":
                    self.infer_to_select(child, end_fields)

                if op_name == 'Filter':
                    self.infer_to_filter(child, end_columns, end_fields)

    def get_field_from_fields(self, columns, fields):
        fields_columns = []
        for f in fields:
            for col in columns:
                if col == self.columns_fields[f]:
                    fields_columns.append(f)
        return fields_columns

    def explain_logical_plan(self):
        print("=" * 20, "\n== Lineage:")
        for op in self.lineage:
            print("{}: {}".format(op, self.operations[op]))

        for i, c in enumerate(self.columns_fields):
            print("Field: ${} - column {} - Dependencies fields: {}"
                  .format(i, c, self.field_dependecies.get(i, [])))

    def infer_to_select(self, child, end_fields):

        new_columns = self.operations[child].current_columns
        self.operations[child].end_columns = new_columns

        new_fields = self.get_field_from_fields(new_columns, end_fields)
        self.operations[child].end_fields = new_fields
        self.operations[child].current_fiedls = new_fields

    def infer_to_dropcolumns(self, child, end_columns, end_fields):
        exclude_fields = self.get_field_from_fields(
                self.operations[child].current_columns, end_fields)

        self.operations[child].end_columns = \
            [col for col in end_columns
             if col not in self.operations[child].current_columns]

        self.operations[child].end_fields = \
            [f for f in self.operations[child].start_fields
             if f not in exclude_fields]

        self.operations[child].current_fields = \
            self.operations[child].end_fields

    def infer_to_map(self, child, end_columns, end_fields):

        # new fields
        new_columns = self.operations[child].current_columns
        new_fields = self.convert_columns2fields(new_columns)
        self.operations[child].current_fields = new_fields

        # mapping new_fields's dependencies
        from ddf_library.columns import col
        function = self.operations[child].settings['function']
        dependency_cols = []
        if isinstance(function, col):
            dependency_cols.append(function.column)
        else:
            for i, a in enumerate(function.args):
                if isinstance(a, col):
                    dependency_cols.append(a.column)

        dependency_fields = self.get_field_from_fields(dependency_cols,
                                                       end_fields)
        self.field_dependecies[new_fields[0]] = dependency_fields

        # update end_columns
        self.operations[child].end_columns = \
            [f for f in end_columns if f not in new_columns] + new_columns

        # update end_fields
        overwrited_fields = self.get_field_from_fields(new_columns, end_fields)
        end_fields = [f for f in end_fields
                      if f not in overwrited_fields] + new_fields
        self.operations[child].end_fields = end_fields

    def infer_to_filter(self, child, end_columns, end_fields):
        columns = self.operations[child].current_columns
        self.operations[child].end_columns = end_columns

        curr_fields = self.get_field_from_fields(columns, end_fields)
        self.operations[child].end_fields = end_fields
        self.operations[child].current_fields = curr_fields



