from distutils.core import setup
from Cython.Build import cythonize

setup(
    ext_modules = cythonize("functions_naivebayes.pyx")
)


#OPT="-O3 -ffast-math" python setup.py build_ext -i
