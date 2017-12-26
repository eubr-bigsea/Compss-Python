from Cython.Build import cythonize
from distutils.core import setup, Extension

#To Compile: OPT="-O3 -ffast-math" python setup.py build_ext -i

setup(
    name="functions knn",
    ext_modules = cythonize("functions_knn.pyx"),
    )
