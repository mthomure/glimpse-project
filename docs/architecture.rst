Library Architecture
********************

Glimpse is broken into low-level components that provide fundamental objects
(such as :mod:`glimpse.models` and :mod:`prototype learning
<glimpse.prototypes>`), mid-level code for managing :mod:`experiments
<glimpse.experiment>`, and the high-level :mod:`glab <glimpse.glab.api>` API.

Low-Level Code
--------------

In Glimpse, a :ref:`backend <backends>` is an implementation of a set of
low-level filtering operations, such as dot products (convolution) and
radial basis functions (RBFs). Each backend uses a different technique for
implementing these operations, and thus provide different performance
benefits. For example, the :class:`base backend
<glimpse.backends.base_backend>` implements most operations in compiled C++.

A :ref:`model <models>` implements a full hierarchical network, with layer
processing implemented using an arbitrary backend. Functionally, a model
provides a transformation from an image to a feature vector given by the
activity of the highest layer in the model. In the case of an HMAX-like
model, the feature vector may be given by the activity of the C2 layer.

In general, however, this transformation is a mapping between arbitrary
*states* of the network. Thus, given an input state and a description of the
desired result, the transformation emits a corresponding output state.
Notably, this deals with all (transitive) dependencies needed to compute the
output state, provided that these dependencies are eventually satisfied. As
an example, this means that a model with three layers (A, B, and C, in that
order) can be used to compute layer C from any input state containing either
A, B, or both A and B.

A :ref:`worker pool <worker pools>` implements a parallelization strategy
for evaluating an arbitrary model on a set of images. Example strategies
include multi-core processing on a single host, as well as multi-host
processing on a compute cluster.

More information can be found below.

.. toctree::
   :maxdepth: 1

   backends
   models
   pools


Experiments
-----------

A central data structure in Glimpse is the :class:`ExperimentData
<glimpse.experiment.ExperimentData>` class, which records everything needed
to identify how an experiment was conducted and what results were found. The
experimental protocol includes the choice of hierarchical model and its
parameters, the corpus of images, and possibly the training and testing
splits (if chosen manually). The experimental results include the set of S2
prototypes, the features extracted by the model from the images, the
training and testing splits (if chosen automatically), the classifier used,
and its performance on the task.

The :mod:`glimpse.experiment` package implements the :class:`ExperimentData
<glimpse.experiment.ExperimentData>` data structure, as well as functions
that operate on this structure to conduct an experiment. The
:mod:`glimpse.glab <glimpse.glab.api>` package provides high-level,
psuedo-declarative interfaces for specifying and running experiments with a
minimum of programming. The rest of the Glimpse library is composed of the
low-level components needed to run an experiment, such as hierarchical model
implementations in :mod:`glimpse.models`, “worker pools” for parallel
computation in :mod:`glimpse.pools`, and prototype learning algorithms in
:mod:`glimpse.prototypes`.


GLAB API
--------

The glab API is documented extensively in the :ref:`user guide
<user-guide>`, and in the :mod:`API <glimpse.glab.api>` and
:mod:`command-line <glimpse.glab.cli>` references.

