#!/usr/bin/env python3
# -*- coding: utf-8 -*-


__author__ = "Lucas Miguel S Ponce"
__email__ = "lucasmsp@gmail.com"

from .regression_lib.ols_regressor import OrdinaryLeastSquares
from .regression_lib.gd_regressor import GDRegressor

__all__ = ['OrdinaryLeastSquares', 'GDRegressor']
