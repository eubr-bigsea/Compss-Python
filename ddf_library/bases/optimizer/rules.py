#!/usr/bin/env python3
# -*- coding: utf-8 -*-


__author__ = "Lucas Miguel S Ponce"
__email__ = "lucasmsp@gmail.com"


class Rule(object):

    description = ""

"""
- PushSelects: fazer seleções virem primeiro.
- MergeSelects: Merge consecutive Selects into a single conjunctive selection.
- ProjectingJoinToProjectOfJoin Turn ProjectingJoin to Project of a Join. This is useful to take advantage of the column selection optimizations and then remove ProjectingJoin for backends that don't have one. 'ProjectingJoin[$1] => Project[$1](Join)'
- RemoveUnusedColumns: For operators that construct new tuples (e.g., GroupBy or Join), we are guaranteed that any columns from an input tuple that are ignored (neither used internally nor to produce the output columns) cannot be used higher in the query tree.
- PushApply: merges consecutive Apply operations into one Apply, possibly dropping some of the produced columns along the way. makes ProjectingJoin only produce columns that are later read.
"""


class PushSelects(object):

    description = "Push down operations that reduce the number of columns."

    def apply(self, logicalplan):
        serial_operations = ['Select', 'DropColumns']
        others_operations = ['Sample', 'take', 'DropNaColumns', "Distinct"] # 'DropNaRows'

        updated = False
        # pushdown select
        for i, uuid in enumerate(reversed(logicalplan.lineage)):
            curr_op = logicalplan.operations[uuid]

            op_name = curr_op.__class__.__name__
            parents = logicalplan.catalog_tasks.get_task_parents(uuid)
            parents = [p for p in parents if p in logicalplan.lineage]
            move_to_up_node = -1

            if len(parents) == 1 and op_name in serial_operations:
                print("[PushSelects] - checking:", uuid)
                end_fields = curr_op.end_fields
                up = True
                while up:
                    up = False
                    p_uuid = parents[0]
                    parent_op = logicalplan.operations[p_uuid]
                    start_fields_p = parent_op.start_fields
                    if set(end_fields).issubset(set(start_fields_p)):

                        move_to_up_node = p_uuid
                        parents = logicalplan.catalog_tasks\
                            .get_task_parents(p_uuid)
                        parents = [p for p in parents if
                                   p in logicalplan.lineage]
                        if len(parents) == 1:
                            up = True

                # remove last_task to up node
                if move_to_up_node != -1:
                    print("[PushSelects]  - moving to:", move_to_up_node)

                    logicalplan.catalog_tasks \
                        .move_node_to_up(move_to_up_node, uuid)

                    idx_s = logicalplan.lineage.index(move_to_up_node)
                    idx_e = logicalplan.lineage.index(uuid)
                    for u in logicalplan.lineage[idx_s:idx_e].copy():
                        logicalplan.lineage.remove(u)
                        logicalplan.catalog_tasks.remove_node(u)

                    updated = True
                    return logicalplan, updated

        return logicalplan, updated


class PushOperationsUp(object):

    description = "Push up operations."

    def apply(self, logicalplan):
        serial_operations = ['Map']
        updated = False
        # pushup select
        for i, uuid in enumerate(logicalplan.lineage):
            curr_op = logicalplan.operations[uuid]

            op_name = curr_op.__class__.__name__
            children = logicalplan.catalog_tasks.get_task_children(uuid)
            children = [p for p in children if p in logicalplan.lineage]
            move_to_up_node = -1

            if len(children) == 1 and op_name in serial_operations:
                print("[PushOperationsUp] - checking:", uuid)
                curr_field = curr_op.current_fields

                up = True
                while up:
                    up = False
                    c_uuid = children[0]
                    parent_op = logicalplan.operations[c_uuid]

                    dependencies_fields = []
                    for f in parent_op.current_fields:
                        dependencies_fields += \
                            logicalplan.field_dependecies.get(f, [])

                    if not any([[True for c in curr_field if c in dependencies_fields],
                                [True for c in curr_field if c in parent_op.current_fields]
                                ]):

                        move_to_up_node = c_uuid

                        children = logicalplan.catalog_tasks\
                            .get_task_children(c_uuid)
                        children = [p for p in children if
                                    p in logicalplan.lineage]
                        if len(children) == 1:
                            up = True

                # remove last_task to up node
                if move_to_up_node != -1:
                    print("[PushOperationsUp] - moving to:", move_to_up_node)

                    logicalplan.catalog_tasks \
                        .move_node_to_down(move_to_up_node, uuid)

                    idx_s = logicalplan.lineage.index(move_to_up_node)
                    logicalplan.lineage.remove(uuid)
                    logicalplan.lineage.insert(idx_s, uuid)

                    updated = True
                    return logicalplan, updated

        return logicalplan, updated


class PushFilters(object):

    description = "Push down operations that reduce the number of rows."

    def apply(self, logicalplan):
        serial_operations = ['Filter']
        others_operations = ['Sample', 'take', 'DropNaColumns',
                             "Distinct"]  # 'DropNaRows'
        updated = False
        # pushdown filter
        for i, uuid in enumerate(reversed(logicalplan.lineage)):
            curr_op = logicalplan.operations[uuid]

            op_name = curr_op.__class__.__name__
            parents = logicalplan.catalog_tasks.get_task_parents(uuid)
            parents = [p for p in parents if p in logicalplan.lineage]
            move_to_up_node = -1

            if len(parents) == 1 and op_name in serial_operations:
                print("[PushFilters] - checking:", uuid)
                curr_fields = curr_op.current_fields

                up = True
                while up:
                    up = False
                    p_uuid = parents[0]
                    parent_op = logicalplan.operations[p_uuid]
                    start_fields_p = parent_op.start_fields
                    if set(curr_fields).issubset(set(start_fields_p)):

                        move_to_up_node = p_uuid
                        parents = logicalplan.catalog_tasks \
                            .get_task_parents(p_uuid)
                        parents = [p for p in parents if
                                   p in logicalplan.lineage]
                        if len(parents) == 1:
                            up = True

                # remove last_task to up node
                if move_to_up_node != -1:
                    print("[PushFilters]  - moving to:", move_to_up_node)
                    logicalplan.catalog_tasks\
                        .move_node_to_up(move_to_up_node, uuid)

                    idx_s = logicalplan.lineage.index(move_to_up_node)
                    logicalplan.lineage.remove(uuid)
                    logicalplan.lineage.insert(idx_s, uuid)

                    updated = True
                    return logicalplan, updated

        return logicalplan, updated


class MergeSelects(object):

    description = "Merge Selects, Drops, ... and possibly others?"

    def __init__(self):
        pass

    def apply(self, lineage, operations, catalog):
        selected_group = []

        valid_operations = ['Select']
        for op in valid_operations:
            for i, uuid in enumerate(lineage):
                if operations[uuid].__class__.__name__ == op:
                    selected_group.append((i, uuid))

            if len(selected_group) > 1:
                tmp_i, tmp_uuid = selected_group[0]
                uuid_to_remove = []
                for i, uuid in selected_group[1:]:
                    if tmp_i + 1 == i:
                        uuid_to_remove.append(tmp_uuid)
                        tmp_i = i
                        tmp_uuid = uuid

                # now we have a list a uuid to remove. To remove, uuid_parent
                # must have only one out edge and uuid_child only one in edge
                for uuid in uuid_to_remove:
                    out_edges = catalog.get_task_children(uuid)
                    in_edges = catalog.get_task_parents(uuid)
                    n_out = len(out_edges)
                    n_in = len(in_edges)
                    if n_in == n_out == 1:
                        catalog.remove_intermediate_node(in_edges[0], uuid,
                                                         out_edges[0])
                        lineage.remove(uuid)
                        operations.pop(uuid, None)


logical_rules = [
    PushOperationsUp,
    PushSelects,
    PushFilters,
    # MergeSelects,
]