#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Lucas Miguel S Ponce"
__email__ = "lucasmsp@gmail.com"

import datetime
import numpy as np
import pandas as pd


class DataType(object):
    INT = 1
    STRING = 2
    DECIMAL = 3
    TIMESTAMP = 4
    DATE = 5
    ARRAY = 6


_converted_types = {
    DataType.INT: int,
    DataType.DECIMAL: float,
    DataType.ARRAY: list,
    DataType.STRING: np.dtype('O'),  #TODO: check string type
    DataType.TIMESTAMP: datetime.datetime,
    DataType.DATE: datetime.date,
}
