#!/usr/bin/python



from pycompss.api.api import compss_wait_on
from pycompss.api.parameter import INOUT, IN
from pycompss.api.task import task
import pandas as pd
import numpy as np


@task(returns=1)
def task_bundle(data, functions, id_frag):

    for f in functions:
        function, settings = f
        print data.columns
        # Used only in save
        if isinstance(settings, dict):
            settings['id_frag'] = id_frag
        data = function(data, settings)

    return data


@task(returns=1)
def task_count(data):
    """Apply the Transformation operation in each row."""
    return len(data)
