#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Lucas Miguel S Ponce"
__email__ = "lucasmsp@gmail.com"

import datetime
import numpy as np

IntegerType = 'integer'
StringType = 'string'
DecimalType = 'decimal'
TimestampType = 'date/time'
DateType = 'date'
ArrayType = 'array'


_converted_types = {
    IntegerType: int,
    DecimalType: float,
    ArrayType: list,
    StringType: np.dtype('O'),
    TimestampType: datetime.datetime,
    DateType: datetime.date,
}
