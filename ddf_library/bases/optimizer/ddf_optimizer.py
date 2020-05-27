#!/usr/bin/env python3
# -*- coding: utf-8 -*-


__author__ = "Lucas Miguel S Ponce"
__email__ = "lucasmsp@gmail.com"

import ddf_library.bases.optimizer.operations as operations_list
from ddf_library.bases.optimizer.rules import logical_rules
from ddf_library.bases.optimizer.logicalplan import LogicalPlan


class DDFOptimizer(object):

    def __init__(self, lineage, catalog_tasks):

        self.catalog_tasks = catalog_tasks
        self.lineage = lineage
        self.operations = {uuid: catalog_tasks.get_task_operation(uuid)
                           for uuid in self.lineage}

        self.logical_plan = LogicalPlan(catalog_tasks, self.operations, lineage)

    def optimize_logical_plain(self):

        for rules in logical_rules:
            updated = True
            r = rules()
            print("Checking Rule: ", r.__class__.__name__, " - ",
                  r.description)
            while updated:
                self.logical_plan, updated = r.apply(self.logical_plan)
                if updated:
                    self.logical_plan.infer_columns()

    def explain_logical_plan(self):
        self.logical_plan.explain_logical_plan()

