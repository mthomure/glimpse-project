.. _models:

######
Models
######

In Glimpse, the model object defines the network topology, including the
number of layers and the operations used at each layer. When the object is
constructed, it is given a backend implementation and a set of parameters
that control its behavior. Several example models are discussed below.

There are currently two hierarchical models included in the Glimpse project.
Both models specify an HMAX-like network, in which an alternating sequence
of "simple" and "complex" layers gradually build up object specificity while
also building invariance to certain transformations. Specifically, an image
is first :ref:`preprocessed <preprocessing>`, and then filtered with a
layer of S1 units to detect edges at various orientation and scale. The
corresponding response maps are then blurred by replacing each local
neighborhood with its maximum activation. This process is then repeated,
with a layer of S2 units being applied to result of the C1 layer. Here, each
S2 unit is characterized by the input template, or *prototype*, to which it
responds. Given N different prototypes, therefore, the S2 layer will
generate N different response maps per scale. Finally, the C2 layer
summarizes the image by computing the maximum response for each S2 prototype
for any location or scale.

The main difference between the two models is the mechanism by which scaling
is accomplished. In the :class:`viz2 model
<glimpse.models.viz2.model.Model>`, the image is analyzed by a different
battery of Gabors for each scale. At the coursest scale, low-frequency
Gabors are used to find blurred edges that occur over a large area.
Conversely, the finest scale uses high-frequency Gabors that detect detailed
edges over a small area.

The :class:`ml model <glimpse.models.ml.model.Model>`, on the other hand,
uses a scale pyramid approach as used by Mutch & Lowe [1]_. Instead of using
different S1 filters for each scale, this model uses different-sized
versions of the input image. Thus, the course-scale response maps are
computed by applying a battery of Gabors to the original input image.
Response maps for the finest-grained scale use the same battery of Gabors,
but apply them to a resized (shrunken) version of the image.

.. note::

   The :mod:`viz2` model results in a stack of response maps for each scale,
   where the dimensions of the stack are fixed. For example, if the spatial
   extent (i.e., the X-Y size) of the course-scale response map is 100 x 200
   units, then the response map for the finest-grained scale will also have
   a spatial extent of 100 x 200 units. This is contrasted with the
   :mod:`ml` model, in which the course-scale response maps are smaller than
   those for the fine-grained scale. This is caused by the reduced size of
   the input image for course scales.


.. _preprocessing:

Preprocessing
-------------

An initial preprocessing stage, referred to as the *retinal* layer, is used
to 1) remove color information, 2) scale the input image, and 3) enhance
image contrast. Color information is removed according to the ITU-R 601-2
luma transform (see the Image.convert method in the `Python Imaging
Library`_). Optionally, the input image can also be scaled (via bicubic
interpolation), such that its shorter side has a given length. Finally,
image contrast is enhanced by applying the :func:`ContrastEnhance
<glimpse.backends.cython_backend.CythonBackend.ContrastEnhance>` backend
function.

.. _Python Imaging Library: http://www.pythonware.com/library/pil/handbook/image.htm

Model Parameters
----------------

Behavior for each model is controlled by a set of parameters, which are
described below according to the layer they affect. To customize these
parameters, the user should first create a Params object corresponding to
the model class, and then set the desired values. An example using the `ml`
model is shown below:

   >>> from glimpse.models import ml
   >>> params = ml.Params()
   >>> params.num_scales = 8
   >>> params.s1_num_orientations = 16
   >>> m = ml.Model(params = params)

Using the :ref:`glab` interface, this simplifies to:

   >>> params = GetParams()
   >>> params.num_scales = 8
   >>> params.s1_num_orientations = 16

**Preprocessing Options**

Retina Enabled
   Whether to use the retinal stage during preprocessing. (Note that color
   information will always be removed.)

   >>> params.retina_enabled = True

Retina Bias
   The bias term used in the :func:`contrast enhancement <contrast enhancement>`
   function to avoid noise amplificiation.

   >>> params.retina_bias = 1.0

Retina Kernel Width
   Size of the local neighborhood used by the preprocessing function.

   >>> params.retina_kwidth = 15

**S1 and S2 Layer Options**

Beta
   Tuning parameter of the activation function (for Rbf and NormRbf).

   >>> params.s1_beta = 1.0
   >>> params.s2_beta = 5.0

Bias
   Bias term for normalization in the activation function (for NormDotProduct
   and NormRbf operations).

   >>> params.s1_bias = 1.0
   >>> params.s2_bias = 1.0

Kernel Width
   Spatial extent of the local neighborhood.

   >>> params.s1_kwidth = 11
   >>> params.s2_kwidth = 7

Operation
   The form of the activation function (one of DotProduct, NormDotProduct, Rbf,
   or NormRbf). See the set of :ref:`filter operations <filtering operations>`
   supported by the backends.

   >>> params.s1_operation = "NormRbf"
   >>> params.s2_operation = "NormRbf"

Sampling
   The sub-sampling factor used when computing S-unit activation.

   >>> params.s1_sampling = 2
   >>> params.s2_sampling = 2

**S1 Gabor Filter Options**

Number of Orientations
   Number of different Gabor orientations.

   >>> params.s1_num_orientations = 8

Shift Orientations
   Whether Gabors are shifted to avoid lining up with the axes.

   >>> params.s1_shift_orientations = True

Number of Phases
   Number of different phases for the S1 Gabor filters (two phases means
   detecting a black to white transition, and vice versa).

   >>> params.s1_num_phases = 2

Number of Scales
   Number of different scales with which to analyze the image.

   >>> params.num_scales = 4

Scale Factor
   (:mod:`ml` model only) The down-sampling factor used to create course
   representations of the input image.

   >>> params.scale_factor = 2**(1/2)

**C1 and C2 Layer Options**

Kernel Width
   Size of the local neighborhood used in the C-unit pooling function.

   >>> params.c1_kwidth = 5
   >>> params.c2_kwidth = 3

Sampling
   The sub-sampling factor used when computing C-unit activiation.

   >>> params.c1_sampling = 2
   >>> params.c2_sampling = 2

C1 Whiten
   Whether to whiten C1 data. See the :func:`Whiten
   <glimpse.models.viz2.model.Whiten>` function.

   >>> params.c1_whiten = False


References
----------

.. [1] ﻿Mutch, J. & Lowe, D.G., 2008. Object Class Recognition and Localization
   Using Sparse Features with Limited Receptive Fields. International Journal of
   Computer Vision, 80(1), p.45-57.
