from pycompss.api.task import task
from pycompss.api.parameter import FILE_IN, FILE_OUT, COLLECTION_IN

import pandas as pd
import time

from ddf_library.utils import save_stage_file, read_stage_file, generate_info, \
    concatenate_pandas

"""
Currently, there are 6 possibilities of dynamic tasks:
 - stage_0in_0out: 
   Serial tasks starting/ending with reading/writing a file in HDFS.

 - stage_0in_1out: 
   Serial tasks starting from reading a file in HDFS and save in parquet as 
   a partial result. 

 - stage_1in_0out:
   Serial tasks reading a file from common file system or a partial result, 
   and saving at end in HDFS.
   
 - stage_1in_1out:
   Serial tasks reading a file from common file system or a partial result, 
   and saving as a partial result or a file on common file system.
   
 - stage_2in_1out: 
   Last tasks reading two partial results and saving as a partial result or 
   a file on common file system.
   
 - stage_2in_0out: 
   Last tasks reading two partial results and saving on HDFS
 
"""


@task(returns=1)
def stage_0in_0out(data_input, stage, id_frag, data_output):
    """
    Serial tasks starting/ending with reading/writing a file in HDFS.

    :param data_input:
    :param stage:
    :param id_frag:
    :return:
    """
    t1 = time.time()
    if stage[-1][0].__name__ == 'task_save':
        stage[-1][1]['output'] = data_output

    data, info = _bundle(data_input, stage, id_frag)
    t2 = time.time()
    print('[INFO] - Time to process the complete stage: {:.0f} seconds'
          .format(t2 - t1))
    return info


@task(returns=1,  data_output=FILE_OUT)
def stage_0in_1out(data, stage, id_frag, data_output):
    """
    Used to read files from HDFS.

    :param data:
    :param stage:
    :param id_frag:
    :param data_output:
    :return:
    """
    t1 = time.time()

    if stage[-1][0].__name__ == 'task_save':
        stage[-1][1]['output'] = data_output
        data, info = _bundle(data, stage, id_frag)
    else:
        data, info = _bundle(data, stage, id_frag)
        save_stage_file(data_output, data)

    t2 = time.time()
    print('[INFO] - Time to process the complete stage: {:.0f} seconds'
          .format(t2 - t1))
    return info


@task(input_file=FILE_IN, returns=1)
def stage_1in_0out(input_file, stage, id_frag, output_file):
    """
    Will perform most functions with the serial tag. Task has 1 data input
    and return 1 data output with its schema

    :param input_file: Input filepath;
    :param stage: a list with functions and its parameters;
    :param id_frag: Block index
    :param output_file: Output filepath;
    :return:
    """
    t1 = time.time()
    # by using parquet, we can specify each column we want to read
    columns = None
    if stage[0][0].__name__ == 'task_select':
        columns = stage[0][1]['columns']

    if stage[-1][0].__name__ == 'task_save':
        stage[-1][1]['output'] = output_file

    data = read_stage_file(input_file, columns)
    data, info = _bundle(data, stage, id_frag)

    t2 = time.time()
    print('[INFO] - Time to process the complete stage: {:.0f}'.format(t2-t1))
    return info


@task(input_file=FILE_IN, output_file=FILE_OUT, returns=1)
def stage_1in_1out(input_file, stage, id_frag, output_file):
    """
    Will perform most functions with the serial tag. Task has 1 data input
    and return 1 data output with its schema

    :param input_file: Input filepath;
    :param stage: a list with functions and its parameters;
    :param id_frag: Block index
    :param output_file: Output filepath;
    :return:
    """
    t1 = time.time()
    print(stage)
    # by using parquet, we can specify columns to read
    if stage[0][0].__name__ == 'task_select':
        columns = stage[0][1]['columns']
        data = read_stage_file(input_file, columns)
    # if the first task is 'read-many-fs' this means that we are receiving
    # a file different from the parquet
    elif stage[0][0].__name__ != 'task_read_many_fs':
        data = read_stage_file(input_file)
    else:
        data = input_file

    if stage[-1][0].__name__ == 'task_save':
        stage[-1][1]['output'] = output_file
        data, info = _bundle(data, stage, id_frag)
    else:
        data, info = _bundle(data, stage, id_frag)
        save_stage_file(output_file, data)

    t2 = time.time()
    print('[INFO] - Time to process the complete stage: {:.0f}'.format(t2-t1))
    return info


@task(input_file1=FILE_IN, input_file2=FILE_IN, returns=1)
def stage_2in_0out(input_file1, input_file2, stage, id_frag, output_file):
    """
    Executed when the first task has two inputs.

    :param input_file1: Input filepath 1;
    :param input_file2: Input filepath 2;
    :param stage: a list with functions and its parameters;
    :param id_frag: Block index
    :return:
    """
    t1 = time.time()

    if stage[-1][0].__name__ == 'task_save':
        stage[-1][1]['output'] = output_file

    data1 = read_stage_file(input_file1)
    data2 = read_stage_file(input_file2)
    data = [data1, data2]
    data, info = _bundle(data, stage, id_frag)
    t2 = time.time()
    print('[INFO] - Time to process the complete stage: {:.0f}'.format(t2 - t1))
    return info


@task(input_file1=FILE_IN, input_file2=FILE_IN, output_file=FILE_OUT, returns=1)
def stage_2in_1out(input_file1, input_file2, stage,
                   id_frag, output_file):
    """
    Executed when the first task has two inputs.

    :param input_file1: Input filepath 1;
    :param input_file2: Input filepath 2;
    :param stage: a list with functions and its parameters;
    :param id_frag: Block index
    :param output_file: Output filepath;
    :return:
    """
    t1 = time.time()
    data1 = read_stage_file(input_file1)
    data2 = read_stage_file(input_file2)
    data = [data1, data2]

    if stage[-1][0].__name__ == 'task_save':
        stage[-1][1]['output'] = output_file
        data, info = _bundle(data, stage, id_frag)
    else:
        data, info = _bundle(data, stage, id_frag)
        save_stage_file(output_file, data)

    t2 = time.time()
    print('[INFO] - Time to process the complete stage: {:.0f}'.format(t2 - t1))
    return info


@task(data_out=FILE_OUT, args=COLLECTION_IN, returns=1)
def concat_n_pandas(data_out, f, args):
    t_start = time.time()
    dfs = [df for df in args if isinstance(df, pd.DataFrame)]
    dfs = concatenate_pandas(dfs)
    info = generate_info(dfs, f)
    save_stage_file(data_out, dfs)
    t_end = time.time()
    print("[INFO] - Time to process task '{}': {:.0f} seconds"
          .format('concat_n_pandas', t_end - t_start))
    return info


def _bundle(data, stage, id_frag):
    """
    Base method to process each stage.

    :param data: The input data;
    :param stage: a list with functions and its parameters;
    :param id_frag: Block index
    :return: An output data and a schema information
    """
    info = None
    for f, current_task in enumerate(stage):
        function, parameters = current_task
        if isinstance(parameters, dict):
            parameters['id_frag'] = id_frag

        t_start = time.time()
        data, info = function(data, parameters)
        t_end = time.time()
        print("[INFO] - Time to process task '{}': {:.0f} seconds"
              .format(function.__name__, t_end-t_start))

    return data, info


