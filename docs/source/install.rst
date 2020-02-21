

******************************
Installation
******************************

The easiest way to install DDF Library is to install it from PyPI. This is the recommended installation method for most users. The instructions for download and installing from source are also provided.


Installing from PyPI
----------------------

DDF Library can be installed via pip from PyPI::

    pip install ddf-pycompss


Source code
-----------

The source code of DDF Library is available online at `Github <https://github.com/eubr-bigsea/Compss-Python>`_. To check out the latest DDF sources::

    git clone https://github.com/eubr-bigsea/Compss-Python.git

Dependencies
-------------

Besides the PyCOMPSs installation, DDF uses others third-party libraries. The main dependencies can be installed by using the command::


    $ pip install -r requirements.txt

    Pyqtree == 0.24
    matplotlib == 1.5.1
    networkx == 1.11
    numpy == 1.16.0
    pandas == 0.23.4
    pyshp == 1.2.11
    python_dateutil == 2.6.1


Optional Dependencies
~~~~~~~~~~~~~~~~~~~~~

If you want to read and save data from HDFS, you need to install `hdfspycompss <https://pypi.org/project/hdfs-pycompss/>`_ library.






