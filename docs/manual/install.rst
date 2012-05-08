************
Installation
************

Prerequisites
=============

The minimum set of dependencies required to run Glimpse includes:

* `Python <http://python.org/>`_ 2.6 (or later but not Python 3)
* `Numpy <http://numpy.scipy.org/>`_ 1.3.0 (or later)
* `Scipy <http://scipy.org/>`_ 0.7.2 (or later)
* `Python Imaging Library <http://www.pythonware.com/products/pil/>`_ 1.1.7 (or
  later)
* `Enthought Traits <http://code.enthought.com/projects/traits/>`_ 3.4 (or
  later)

This allows Glimpse to function as a feature extractor, but will prohibit any
other uses (e.g., classification of an image based on those features).

Additional functionality (e.g., SVM classifiers, parallel processing)
requires the following packages. (Note that Glimpse may work with previous
versions, but this is untested.)

* `LIBSVM <http://www.csie.ntu.edu.tw/~cjlin/libsvm>`_ with python modules 2.91
  (or later) for image classification
* `libjpeg <http://libjpeg.sourceforge.net/>`_ 8 (or later) for I/O of
  JPEG-formatted images in Python Imaging Library
* `Enthought Traits GUI <http://code.enthought.com/projects/traits_gui/>`_ 3.4
  (or later) for graphical editing of model parameters
* `Gearman library <http://gearman.org/>`_ 0.14 (or later) for cluster-based
  feature extraction
* `pyzmq <http://zeromq.github.com/pyzmq/>`_ 2.1.11 (or later) for cluster-based
  feature extraction
* `Cython <http://cython.org/>`_ 0.12 (or later) to modify C++ or Cython code

Additionally, the following packages are recommended for working with Glimpse.

* `IPython <http://ipython.org/>`_
* `matplotlib <http://matplotlib.sourceforge.net/>`_

Installing on Ubuntu/Debian Linux
---------------------------------

On Ubuntu or Debian Linux, you can use the APT package manager.
To install all dependencies, type::

   > sudo apt-get install python python-dev python-setuptools \
     python-numpy python-scientific python-imaging libjpeg8 \
     python-traits python-traitsgui libsvm2 python-libsvm \
     gearman-job-server gearman-tools zmq pyzmq

Installing Glimpse
==================

To build Glimpse, just unpack it and type::

   > python setup.py install

Alternatively, the project can be installed from the Python Package Index with::

   > pip install glimpse
