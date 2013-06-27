.. _backends:

########
Backends
########

A backend is an implementation of a consistent interface, which provides
basic operations for filtering N-dimensional arrays. These include
:ref:`filtering <filtering operations>` operations that build selectivity,
:ref:`pooling <pooling operations>` operations that build invariance, and an
operation providing local :ref:`contrast enhancement <contrast enhancement>`
of an image.


.. _filtering operations:

Filtering
---------

Four filter operations are supported. The operation :meth:`DotProduct
<glimpse.backends.cython_backend.CythonBackend.DotProduct>` compares the
input neigborhood and the weight vector (i.e., prototype) using a dot
product, where each output is given by

.. math::
   y = X^T W

for input neighborhood :math:`X` (given as a vector) and weight vector
:math:`W`, where :math:`X^T` denotes the matrix transpose. The operation
:meth:`NormDotProduct
<glimpse.backends.cython_backend.CythonBackend.NormDotProduct>` is similar,
but constrains each vector to have unit norm. Thus, the output is given by

.. math::
   y = \text{NDP}(X, W) = \frac{X^T W}{\left\Vert X \right\Vert \left\Vert W \right\Vert} \, ,

where :math:`\left\Vert \cdot \right\Vert` denotes the Euclidean norm.

Instead of a dot product, the operation :meth:`Rbf
<glimpse.backends.cython_backend.CythonBackend.Rbf>` compares the input and
weight vectors using a radial basis function (RBF). Here, the output is
given as

.. math::
   y = \exp \left\{ - \beta \left\Vert X - W \right\Vert ^2 \right\} \, ,

where :math:`\beta` controls the sensitivity of the RBF. Constraining the
vector norm of the arguments gives the final operation :meth:`NormRbf
<glimpse.backends.cython_backend.CythonBackend.NormRbf>`, where the output is
given as

.. math::
   y = \exp \left\{ - 2\beta \left(1 - \text{NDP}(X, W) \right) \right\} \, ,

Here, we have used the bilinearity of the inner product to write the
distance as

.. math::
   \left\Vert V_a - V_b \right\Vert ^2 = 2 - 2 V_a^T V_b

for unit vectors :math:`V_a` and :math:`V_b`.


.. _pooling operations:

Pooling
-------

Currently, the only operation that is supported is a maximum-value pooling
function. For a local neighborhood of the input :math:`X`, this computes an
output value as

.. math::
   y = max_{i,j} \ x_{ij} \ .

This has been argued to provide a good match to cortical response properties
[1]_, and has been shown in practice to lead to better performance [2]_.


.. _contrast enhancement:

Contrast Enhancement
--------------------

Given a local input neighborhood :math:`X`, the output is

.. math::
   y = \frac{x_c - \mu}{\max(\sigma, \epsilon)}

where :math:`x_c` is the center of the input neighborhood, :math:`\mu` and
:math:`\sigma` are the mean and standard deviation of :math:`X`, and
:math:`\epsilon` is a bias term. This term is used to avoid the
amplificiation of noise and to ensure a non-zero divisor.


References
----------

.. [1] ﻿Serre, T., Oliva, A. & Poggio, T., 2007. A feedforward architecture
   accounts for rapid categorization. Proceedings of the National Academy of
   Sciences, 104(15), p.6424-6429.

.. [2] ﻿Boureau, Y.-L. et al., 2010. Learning mid-level features for
   recognition. In *Computer Vision and Pattern Recognition 2010*. IEEE, pp.
   2559-2566.
