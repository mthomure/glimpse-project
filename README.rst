========================================================
GLIMPSE - The General Layer-wise IMage ProceSsing Engine
========================================================

The GLIMPSE project is a library for implementing hierarchical visual models in
C++ and Python. The goal of this project is to allow a broad range of
feed-forward, hierarchical models to be encoded in a high-level declarative
manner, with low-level details of the implementation hidden from view. GLIMPSE
combines an efficient implementation with the ability to leverage parallel
processing facilities and is designed to run on multiple operating systems using
only common, freely-available components. A prototype of GLIMPSE has been used
to encode an HMAX-like model, achieving results comparable with those found in
the literature.

The GLIMPSE library began as a port of the Petascale Artificial Neural Network
project [1]_, and owes a great deal to that work. Additionally, we acknowledge
NSF Grant 1018967 (PIs: Melanie Mitchell and Garrett Kenyon) for support.

See the `user guide`_ for more information.

.. [1] S. Brumby, G. Kenyon, W. Landecker, C. Rasmussen, S. Swaminarayan,
   and L. Bettencourt, "Large-Scale Functional Models of Visual Cortex for
   Remote Sensing," in *Applied Imagery Pattern Recognition 2009 (AIPR ’09)*,
   2009.

.. _user guide: https://docs.google.com/document/pub?id=1mhrX7jSVR_Ur9XE-DpalH2UYDqbalHKHZgoGrEcvTT4
