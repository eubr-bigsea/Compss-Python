import sys
import setuptools


def get_requirements():
    """
    lists the requirements to install.
    """
    try:
        with open('requirements.txt') as f:
            requirements = f.read().splitlines()
    except Exception as ex:
        requirements = []
    return requirements


def get_readme():
    try:
        with open('README.md') as f:
            readme = f.read()
    except Exception as ex:
        readme = ''
    return readme


if sys.version_info < (3, 6):
    print("Python versions prior to 3.6 are not supported for DDF Library.",
          file=sys.stderr)
    sys.exit(-1)

try:
    exec(open('ddf_library/version.py').read())
except IOError:
    print("Failed to load DDF version file for packaging. You must be in "
          "DDF's python dir.", file=sys.stderr)
    sys.exit(-1)

VERSION = __version__

setuptools.setup(
     name='ddf-pycompss',  
     version=VERSION,
     author="Lucas Miguel Ponce",
     author_email="lucasmsp@dcc.ufmg.br",
     description="A PyCOMPSs library for Big Data scenarios.",
     url="https://github.com/eubr-bigsea/Compss-Python",
     license='http://www.apache.org/licenses/LICENSE-2.0',
     platforms=['Linux'],
     long_description=get_readme(),
     long_description_content_type='text/markdown',
     include_package_data=True,
     classifiers=[
         'Programming Language :: Python :: 3',
         "Programming Language :: Python :: 3.6",
         'License :: OSI Approved :: Apache Software License',
         "Operating System :: POSIX :: Linux",
         "Topic :: Software Development :: Libraries",
         "Topic :: Software Development :: Libraries :: Python Modules",
         "Topic :: System :: Distributed Computing",
     ],
     packages=setuptools.find_packages(),
     install_requires=get_requirements(),

 )



