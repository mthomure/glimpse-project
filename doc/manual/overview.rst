################
Library Overview
################

The basic parts of the Glimpse library include backends, models, and worker
pools. Additionally, Glimpse supports a number of simple algorithms for
learning filter kernels (i.e., prototypes).

.. toctree::
   :maxdepth: 1

   backends
   models
   pools/index
   learning

In Glimpse, a :ref:`backend <backends>` is an implementation of a set of
low-level filtering operations, such as dot products (convolution) and
radial basis functions (RBFs). Each backend uses a different technique for
implementing these operations, and thus provide different performance
benefits. For example, :class:`CythonBackend
<glimpse.backends.cython_backend.CythonBackend>` implements most operations
in compiled C++, and is thus the default backend.

A :ref:`model <models>` implements a full hierarchical network, with layer
processing implemented using an arbitrary backend. Functionally, a model
provides a transformation from an image to a feature vector given by the
activity of the highest layer in the model. For example, in the case of an
HMAX-like model [1]_, the feature vector is given by the activity of the C2
layer.

In general, however, this transformation is a mapping between arbitrary
*states* of the network. Thus, given an input state and a description of the
desired result, the transformation emits a corresponding output state.
Notably, this deals with all (transitive) dependencies needed to compute the
output state, provided that these dependencies are eventually satisfied. As
an example, this means that a model with three layers (A, B, and C, in that
order) can be used to compute layer C from any input state containing either
A, B, or both A and B.

Finally, a :ref:`worker pool <worker pools>` implements a parallelization
strategy for evaluating an arbitrary model on a set of images. Example
strategies include multi-core processing on a single host, as well as
multi-host processing on a compute cluster.


References
----------

.. [1] ﻿Serre, T., Oliva, A. & Poggio, T., 2007. A feedforward architecture
   accounts for rapid categorization. Proceedings of the National Academy of
   Sciences, 104(15), p.6424-6429.
