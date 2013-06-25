************
Installation
************

Prerequisites
=============

The minimum set of dependencies required to run Glimpse includes:

* `Python <http://python.org/>`_ 2.6 (or later, but not Python 3)
* `Numpy <http://numpy.scipy.org/>`_ 1.3.0 (or later)
* `Scipy <http://scipy.org/>`_ 0.7.2 (or later)
* `Python Imaging Library <http://www.pythonware.com/products/pil/>`_ 1.1.7 (or
  later)
* `Enthought Traits <http://code.enthought.com/projects/traits/>`_ 3.4 (or
  later)
* `Enthought Traits GUI <http://code.enthought.com/projects/traits_gui/>`_ 3.4
  (or later) for graphical editing of model parameters
* `Scikit-Learn <http://scikit-learn.org/>`_ machine learning library.
* `decorator <https://pypi.python.org/pypi/decorator>`_ library.
* `Matplotlib <http://matplotlib.sourceforge.net/>`_ for plotting.

Additional packages that are recommended or that provide extended behavior
include:

* `IPython <http://ipython.org/>`_  is highly recommended for interactive work,
  and  required to use a compute cluster for feature extraction.
* `Cython <http://cython.org/>`_ 0.12 (or later) to modify C++ or Cython code.
* `libjpeg <http://libjpeg.sourceforge.net/>`_ 8 (or later) for I/O of
  JPEG-formatted images in Python Imaging Library. This is often installed
  already, but OS X users will have to install it manually.

Installing Prerequisites on Ubuntu/Debian Linux
-----------------------------------------------

On Ubuntu or Debian Linux, you can use the APT package manager.
To install all dependencies, type::

   $ sudo apt-get install python python-dev python-setuptools \
     python-numpy python-scientific python-imaging libjpeg8 \
     python-traits python-traitsgui ipython python-matplotlib python-decorator

Installing Glimpse
==================

To build Glimpse, type::

   $ pip install glimpse

On Mac OSX, you may need build for a 32-bit architecture. For example, this
happens when using 32-bit Python on a 64-bit machine. To do this, download
and unpack the project, and then use the modified install command::

   $ ARCHFLAGS='-arch i386' python setup.py install
