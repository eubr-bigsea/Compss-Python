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


setuptools.setup(
     name='ddf-pycompss',  
     version='0.4',
     author="Lucas Miguel Ponce",
     author_email="lucasmsp@dcc.ufmg.br",
     summary="A PyCOMPSs library for Big Data scenarios.",
     description="A PyCOMPSs library for Big Data scenarios.",
     url="https://github.com/eubr-bigsea/Compss-Python",
     license='Apache License, Version 2.0',
     platforms=['Linux'],
     long_description=get_readme(),
     long_description_content_type='text/markdown',
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



