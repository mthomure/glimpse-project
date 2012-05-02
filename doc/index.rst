#######
Glimpse
#######

Welcome to Glimpse, a General Layer-wise IMage ProceSsing Engine!

.. toctree::
   :maxdepth: 1

   manual/install
   manual/quick-start
   manual/overview
   manual/glab
   manual/command-line/index
   manual/future
   API Reference <ref/index.rst>

The Glimpse project is a library for implementing hierarchical visual models
in C++ and Python. The goal of this project is to allow a broad range of
feed-forward, hierarchical models to be encoded in a high-level declarative
manner, with low-level details of the implementation hidden from view. This
project combines an efficient implementation with the ability to leverage
parallel processing facilities and is designed to run on multiple operating
systems using only common, freely-available components. A prototype of
Glimpse has been used to encode an HMAX-like model, achieving results
comparable with those found in the literature.

Source code for the project can be found on `github
<https://github.com/mthomure/glimpse-project>`_ and the `Python Package
Index (PyPI) <http://pypi.python.org/pypi/glimpse>`_.


Acknowledgements
----------------

This project began as a port of the Petascale Artificial Neural Network
project [1]_, and owes a great deal to that work. Additionally, we
acknowledge NSF Grant 1018967 (PIs: Melanie Mitchell and Garrett Kenyon) for
support.


References
----------

.. [1] S. Brumby, G. Kenyon, W. Landecker, C. Rasmussen, S. Swaminarayan,
   and L. Bettencourt, "Large-Scale Functional Models of Visual Cortex for
   Remote Sensing," in *Applied Imagery Pattern Recognition 2009 (AIPR ’09)*,
   2009.
