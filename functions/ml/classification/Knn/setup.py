from distutils.core import setup
from Cython.Build import cythonize

#To Compile: OPT="-O3 -ffast-math" python setup.py build_ext -i

setup(
    ext_modules = cythonize("functions_knn.pyx")
)
