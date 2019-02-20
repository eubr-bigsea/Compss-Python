import setuptools

setuptools.setup(
     name='ddf-pycompss',  
     version='0.2',
     author="Lucas Miguel Ponce",
     author_email="lucasmsp@dcc.ufmg.br",
     summary="A PyCOMPSs library for Big Data scenarios.",
     description="A PyCOMPSs library for Big Data scenarios.",
     url="https://github.com/eubr-bigsea/Compss-Python",
     license='Apache License, Version 2.0',
     platforms=['Linux'],
     long_description="""

     The Distributed DataFrame Library provides distributed algorithms and operations ready to use as a library implemented over PyCOMPSs programming model. Currently, it is highly focused on ETL (extract-
     transform-load) and Machine Learning algorithms to Data Science tasks. DDF is greatly inspired by Spark's DataFrame and its operators.

     Currently, an operation can be of two types, transformations or actions. Action operations are those that produce a final result (whether to save to a file or to display on screen). Transformation
     operations are those that will transform an input DDF into another output DDF. Besides this classification, there are operations with one processing stage and those with two or more stages of processing (those that
     need to exchange information between the partitions).

     When running DDF operation/algorithms, a context variable (COMPSs Context) will check the possibility of optimizations during the scheduling of COMPS tasks. These optimizations can be of the type: grouping one 
     stage operations to a single task COMPSs and stacking operations until an action operation is found.
    """,
     classifiers=[
         "Programming Language :: Python :: 2.7",
         'License :: OSI Approved :: Apache Software License',
         "Operating System :: POSIX :: Linux",
         "Topic :: Software Development :: Libraries",
         "Topic :: Software Development :: Libraries :: Python Modules",
         "Topic :: System :: Distributed Computing",
     ],
    packages=setuptools.find_packages(),
    install_requires=[
        "Pyqtree>=0.24",
        "matplotlib>=1.5.1",
        "networkx>=1.11",
        "numpy>=1.16.0",
        "pandas>=0.23.4",
        "pyshp>=1.2.11",
        "python_dateutil>=2.6.1",
    ],

 )



